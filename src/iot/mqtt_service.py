from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import paho.mqtt.client as mqtt
import pytesseract
from rich import print as rprint

from src.models.model import load_model


@dataclass
class MQTTConfig:
    broker: str = "localhost"
    port: int = 1883
    username: Optional[str] = None
    password: Optional[str] = None
    sub_image_topic: str = "esp32/essay_image"
    sub_text_topic: str = "esp32/essay_text"
    pub_result_topic: str = "esp32/essay_result"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def preprocess_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 10)
    return thr


def ocr_to_text(img_bgr: np.ndarray) -> str:
    proc = preprocess_for_ocr(img_bgr)
    text = pytesseract.image_to_string(proc, lang="eng")
    return text.strip()


def decode_image(payload: bytes) -> Optional[np.ndarray]:
    try:
        # Try raw JPEG/PNG bytes first
        arr = np.frombuffer(payload, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass
    try:
        # Try base64-encoded image
        decoded = base64.b64decode(payload)
        arr = np.frombuffer(decoded, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


class EssayMQTTService:
    def __init__(self, model_path: Path, cfg: MQTTConfig, tesseract_cmd: Optional[str] = None):
        self.model = load_model(model_path)
        self.cfg = cfg
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        if cfg.username and cfg.password:
            self.client.username_pw_set(cfg.username, cfg.password)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    def start(self) -> None:
        self.client.connect(self.cfg.broker, self.cfg.port, keepalive=60)
        self.client.loop_start()
        # Subscribe after connection established in on_connect
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            rprint("Stopping MQTT service...")
        finally:
            self.client.loop_stop()
            self.client.disconnect()

    # MQTT callbacks
    def on_connect(self, client, userdata, flags, reason_code, properties=None):
        rprint(f"Connected to MQTT broker with code {reason_code}")
        client.subscribe(self.cfg.sub_image_topic, qos=1)
        client.subscribe(self.cfg.sub_text_topic, qos=1)
        rprint(f"Subscribed to {self.cfg.sub_image_topic} and {self.cfg.sub_text_topic}")

    def publish_result(self, source: str, text: str, pred: int):
        label = "AI" if int(pred) == 1 else "Human"
        payload = {
            "time": now_iso(),
            "source": source,
            "label": label,
            "pred": int(pred),
            "excerpt": (text[:240] + "...") if len(text) > 240 else text,
        }
        self.client.publish(self.cfg.pub_result_topic, json.dumps(payload), qos=1)

    def classify_text(self, text: str) -> int:
        return int(self.model.predict([text])[0])

    def on_message(self, client, userdata, msg):
        topic = msg.topic
        try:
            if topic == self.cfg.sub_text_topic:
                # Accept plain text or JSON {"text": "..."}
                try:
                    obj = json.loads(msg.payload.decode("utf-8", errors="ignore"))
                    text = obj.get("text", "") if isinstance(obj, dict) else str(obj)
                except Exception:
                    text = msg.payload.decode("utf-8", errors="ignore")
                if not text.strip():
                    return
                pred = self.classify_text(text)
                self.publish_result("text", text, pred)
                rprint({"topic": topic, "pred": pred})
            elif topic == self.cfg.sub_image_topic:
                img = decode_image(msg.payload)
                if img is None:
                    rprint("[yellow]Received image payload but failed to decode[/yellow]")
                    return
                text = ocr_to_text(img)
                if not text:
                    rprint("[yellow]OCR produced empty text[/yellow]")
                    return
                pred = self.classify_text(text)
                self.publish_result("image", text, pred)
                rprint({"topic": topic, "len_text": len(text), "pred": pred})
        except Exception as e:
            rprint(f"[red]Error handling message on {topic}: {e}[/red]")


def run_service(
    broker: str,
    port: int,
    sub_image_topic: str,
    sub_text_topic: str,
    pub_result_topic: str,
    model_path: Path,
    username: Optional[str] = None,
    password: Optional[str] = None,
    tesseract_cmd: Optional[str] = None,
):
    cfg = MQTTConfig(
        broker=broker,
        port=port,
        username=username,
        password=password,
        sub_image_topic=sub_image_topic,
        sub_text_topic=sub_text_topic,
        pub_result_topic=pub_result_topic,
    )
    svc = EssayMQTTService(model_path=model_path, cfg=cfg, tesseract_cmd=tesseract_cmd)
    svc.start()


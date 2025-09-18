#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <PubSubClient.h>

// Fill in your WiFi and MQTT settings below or via secrets.h
#if __has_include("secrets.h")
#include "secrets.h"
#else
// WiFi
const char* WIFI_SSID = "YOUR_SSID";
const char* WIFI_PASS = "YOUR_PASSWORD";
// MQTT
const char* MQTT_HOST = "192.168.1.100"; // broker IP or host
const uint16_t MQTT_PORT = 1883;
const char* MQTT_USER = nullptr; // optional
const char* MQTT_PASS = nullptr; // optional
#endif

// Topics
const char* TOPIC_TEXT   = "esp32/essay_text";    // publish text
const char* TOPIC_RESULT = "esp32/essay_result";  // (optional) where PC publishes results back

WiFiClient wifiClient;
PubSubClient mqtt(wifiClient);

// ESPNOW message (very simple text demo)
typedef struct __attribute__((packed)) {
  uint8_t msg_id;
  char text[200];
} now_msg_t;

void onEspNowRecv(const uint8_t * mac, const uint8_t *incomingData, int len) {
  if (len <= 0) return;
  // For demo: treat payload as text (ensure null-terminated)
  String text;
  text.reserve(len + 1);
  for (int i = 0; i < len; ++i) text += (char)incomingData[i];
  // Publish to MQTT as plain text
  mqtt.publish(TOPIC_TEXT, text.c_str());
}

void ensureMqtt() {
  while (!mqtt.connected()) {
    String clientId = String("esp32-gateway-") + String((uint32_t)ESP.getEfuseMac(), HEX);
    if (MQTT_USER && MQTT_PASS) {
      if (mqtt.connect(clientId.c_str(), MQTT_USER, MQTT_PASS)) break;
    } else {
      if (mqtt.connect(clientId.c_str())) break;
    }
    delay(1000);
  }
}

void setup() {
  Serial.begin(115200);
  delay(100);
  Serial.println("\n[Gateway] Booting...");

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("[Gateway] WiFi connecting");
  int tries = 0;
  while (WiFi.status() != WL_CONNECTED && tries < 60) {
    delay(500);
    Serial.print(".");
    tries++;
  }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("[Gateway] WiFi connected: %s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println("[Gateway] WiFi failed, continuing for ESPNOW->Serial mode");
  }

  // ESPNOW init
  if (esp_now_init() == ESP_OK) {
    esp_now_register_recv_cb(onEspNowRecv);
    Serial.println("[Gateway] ESP-NOW ready");
  } else {
    Serial.println("[Gateway] ESP-NOW init failed");
  }

  // MQTT setup
  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  if (WiFi.status() == WL_CONNECTED) {
    ensureMqtt();
    Serial.println("[Gateway] MQTT connected");
  }

  Serial.println("[Gateway] Type 'TEXT: your text here' in Serial to publish");
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    if (!mqtt.connected()) ensureMqtt();
    mqtt.loop();
  }

  // Simple Serial bridge: send lines prefixed with TEXT:
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.startsWith("TEXT:")) {
      String payload = line.substring(5);
      payload.trim();
      if (payload.length() > 0 && mqtt.connected()) {
        mqtt.publish(TOPIC_TEXT, payload.c_str());
        Serial.println("[Gateway] Published line to MQTT");
      } else {
        Serial.println("[Gateway] Not published (empty or MQTT down)");
      }
    }
  }

  delay(5);
}


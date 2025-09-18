 Human vs AI Essay Classifier

Binary classifier to distinguish AI-generated vs human-written essays using the Kaggle dataset `balanced_ai_human_prompts.csv`. Built with scikit-learn and managed with uv for reproducible environments.

 Team Members

|  AC.NO   | Name                 | Role    | Contributions        |
|----------|----------------------|---------|----------------------|
| 202174119| Amr Tarek Al-Hammadi | Lead    | Data prep, modeling  |
|                                 | Analyst | EDA, visualization   |
|                                 | Engineer| Evaluation, packaging|

 Installation and Setup

powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

Prerequisites
- Python 3.12+
- uv (https://docs.astral.sh/uv/) or install using the following command in your terminal:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
```powershell
python -m pip install "setuptools<58"
python -m pip install uv
```

Steps
- Clone or open this folder.
- Install dependencies with uv:
  - `uv sync`
- Optional: activate the venv for your shell
  - `uv venv --python 3.12`
  - `uv run python --version`

 Project Structure

```
project/
- README.md
- pyproject.toml
- .python-version
- main.py               # CLI entry point
- src/
  - data/
    - dataset.py        # Load/split utilities
  - models/
    - model.py          # Pipeline, train/eval
  - utils/
    - io.py             # Save/load model
    - metrics.py        # Metrics helpers
- notebooks/
- data/
  - balanced_ai_human_prompts.csv  # Dataset
- docs/
```

 Usage

Train (uses the CSV in `data/` by default):
- `uv run python main.py train --data data/balanced_ai_human_prompts.csv --model-path models/model.joblib`

Evaluate on the test split:
- `uv run python main.py evaluate --data data/balanced_ai_human_prompts.csv --model-path models/model.joblib`
- Exports to `docs/`: `confusion_matrix.png`, `classification_report.txt`, and `metrics.json`.

Predict a single text input:
- `uv run python main.py predict --model-path models/model.joblib --text "Your essay text here..."`

Hyperparameter tuning (GridSearchCV over TF-IDF + LinearSVC):
- `uv run python main.py tune --data data/balanced_ai_human_prompts.csv --cv 3 --jobs -1 --model-path models/model_tuned.joblib`

Run the EDA notebook:
- `uv run jupyter notebook notebooks/01_eda.ipynb`

Launch GUI to paste text and get predictions:
- `uv run python main.py gui`

## ESP32 + MQTT + OpenCV OCR Integration

Architecture
- ESP32-CAM (or camera source) captures an essay image.
- ESP-NOW to a gateway ESP32, which forwards the data over Wi‑Fi using MQTT.
- This project runs an MQTT service that:
  - Subscribes to `esp32/essay_image` (JPEG base64 or raw bytes) or `esp32/essay_text` (UTF‑8 text/JSON).
  - If image: uses OpenCV preprocessing + Tesseract OCR to extract text.
  - Classifies with the trained model and publishes a result JSON to `esp32/essay_result`.

Run the service
- `uv run python main.py iot --broker localhost --port 1883 --model-path models/model.joblib`
- Optional if Tesseract is not on PATH: `--tesseract-cmd "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"`

MQTT topics (defaults)
- Subscribe images: `esp32/essay_image` (payload: raw JPEG bytes or base64 string)
- Subscribe text: `esp32/essay_text` (payload: plain text or `{ "text": "..." }`)
- Publish results: `esp32/essay_result` (JSON: `{ time, source, label, pred, excerpt }`)

ESP32 sketch outline
- Device A (ESP32-CAM): capture JPEG, send via ESP‑NOW in chunks (<=200 bytes) with seq numbers.
- Device B (ESP32 Wi‑Fi gateway): receive chunks, reassemble JPEG, then `MQTT.publish("esp32/essay_image", base64)`.
- For simpler integration, you can skip images and just send text lines from ESP‑NOW to the gateway, and publish to `esp32/essay_text`.

External dependency for OCR
- Install Tesseract on the machine running this service (Windows installer adds Tesseract; on Ubuntu: `sudo apt install tesseract-ocr`).
- Python packages are handled by uv (opencv-python + pytesseract).

### ESP32 Gateway Upload (Arduino IDE)
- Board: ESP32 Dev Module (or your specific ESP32 board)
- Port: COM5 (detected)
- Open `esp32/gateway_mqtt_bridge/gateway_mqtt_bridge.ino` in Arduino IDE
- Install ESP32 core (Boards Manager: add `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`, then install "esp32")
- Library: install `PubSubClient` (Library Manager)
- Create `esp32/gateway_mqtt_bridge/secrets.h` by copying `secrets.h.example` and fill WiFi/MQTT
- Select Port: COM5
- Upload

Quick test via Serial
- Open Serial Monitor (115200 baud)
- Type: `TEXT: This is a sample essay line...`
- The gateway publishes to `esp32/essay_text`; the Python service will classify and publish to `esp32/essay_result`.

Notes
- The script creates a stratified train/test split (80/20) on-the-fly from the provided CSV. The label column is `generated` ("1" = AI-generated, "0" = human).
- The model is a scikit-learn pipeline: TF-IDF (word + bigram) + Linear SVM.

 Results

After training, the script prints accuracy, precision, recall, and F1 on the test set and saves the model to `models/model.joblib`.

 Contributing

- Create a feature branch: `git checkout -b feature-name`
- Commit with clear messages and keep changes focused.
- Open a pull request for review.

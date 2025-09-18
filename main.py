import argparse
from pathlib import Path
from rich import print as rprint

from src.data.dataset import load_dataset, train_test_split_stratified
from src.models.model import (
    build_pipeline,
    train_pipeline,
    evaluate_pipeline,
    save_model,
    load_model,
    tune_pipeline,
)
from src.app.gui import run_gui
from src.iot.mqtt_service import run_service as run_mqtt_service
from src.utils.metrics import (
    gen_confusion_matrix,
    gen_classification_report_text,
    plot_confusion_matrix,
    save_classification_report,
    save_metrics_json,
)


def cmd_train(args: argparse.Namespace) -> None:
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = load_dataset(data_path)
    X_train, X_test, y_train, y_test = train_test_split_stratified(df, test_size=args.test_size, random_state=args.seed)

    model = build_pipeline()
    model = train_pipeline(model, X_train, y_train)

    metrics = evaluate_pipeline(model, X_test, y_test)
    rprint({
        "accuracy": round(metrics["accuracy"], 4),
        "precision": round(metrics["precision"], 4),
        "recall": round(metrics["recall"], 4),
        "f1": round(metrics["f1"], 4),
    })

    save_model(model, Path(args.model_path))
    rprint(f"Saved model to {args.model_path}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    model = load_model(Path(args.model_path))
    df = load_dataset(data_path)
    _, X_test, _, y_test = train_test_split_stratified(df, test_size=args.test_size, random_state=args.seed)

    metrics = evaluate_pipeline(model, X_test, y_test)
    rprint({
        "accuracy": round(metrics["accuracy"], 4),
        "precision": round(metrics["precision"], 4),
        "recall": round(metrics["recall"], 4),
        "f1": round(metrics["f1"], 4),
    })

    # Exports: confusion matrix image and classification report text/json under docs/
    preds = model.predict(X_test)
    cm = gen_confusion_matrix(y_test, preds)
    labels = ["Human", "AI"]
    plot_confusion_matrix(cm, labels, Path("docs/confusion_matrix.png"))

    report_text = gen_classification_report_text(y_test, preds, target_names=labels)
    save_classification_report(report_text, Path("docs/classification_report.txt"))
    save_metrics_json(metrics, Path("docs/metrics.json"))
    rprint("Saved evaluation artifacts to docs/ (confusion_matrix.png, classification_report.txt, metrics.json)")


def cmd_predict(args: argparse.Namespace) -> None:
    model = load_model(Path(args.model_path))
    text = args.text
    pred = model.predict([text])[0]
    label = "AI" if int(pred) == 1 else "Human"
    rprint({"pred": int(pred), "label": label})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Human vs AI essay classifier")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train
    p_train = subparsers.add_parser("train", help="Train model and save to disk")
    p_train.add_argument("--data", type=str, default="data/balanced_ai_human_prompts.csv", help="Path to CSV dataset")
    p_train.add_argument("--model-path", type=str, default="models/model.joblib", help="Output path for model")
    p_train.add_argument("--test-size", type=float, default=0.2, help="Test size fraction")
    p_train.add_argument("--seed", type=int, default=42, help="Random seed")
    p_train.set_defaults(func=cmd_train)

    # Evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate saved model on test split")
    p_eval.add_argument("--data", type=str, default="data/balanced_ai_human_prompts.csv", help="Path to CSV dataset")
    p_eval.add_argument("--model-path", type=str, default="models/model.joblib", help="Path to saved model")
    p_eval.add_argument("--test-size", type=float, default=0.2, help="Test size fraction")
    p_eval.add_argument("--seed", type=int, default=42, help="Random seed")
    p_eval.set_defaults(func=cmd_evaluate)

    # Predict
    p_pred = subparsers.add_parser("predict", help="Predict a single text string")
    p_pred.add_argument("--model-path", type=str, default="models/model.joblib", help="Path to saved model")
    p_pred.add_argument("--text", type=str, required=True, help="Text to classify")
    p_pred.set_defaults(func=cmd_predict)

    # Tune
    def cmd_tune(args: argparse.Namespace) -> None:
        data_path = Path(args.data)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        df = load_dataset(data_path)
        X = df["text"].astype(str).tolist()
        y = df["generated"].astype(int).tolist()
        best_est, best_params, best_score = tune_pipeline(
            X, y,
            param_grid=None,
            cv=args.cv,
            n_jobs=args.jobs,
            verbose=1,
        )
        rprint({"best_f1_cv": round(best_score, 4), "best_params": best_params})
        if args.model_path:
            save_model(best_est, Path(args.model_path))
            rprint(f"Saved tuned model to {args.model_path}")

    p_tune = subparsers.add_parser("tune", help="Grid search hyperparameters and optionally save best model")
    p_tune.add_argument("--data", type=str, default="data/balanced_ai_human_prompts.csv", help="Path to CSV dataset")
    p_tune.add_argument("--cv", type=int, default=3, help="Cross-validation folds")
    p_tune.add_argument("--jobs", type=int, default=-1, help="Parallel jobs for grid search")
    p_tune.add_argument("--model-path", type=str, default="models/model_tuned.joblib", help="Output path for best model")
    p_tune.set_defaults(func=cmd_tune)

    # GUI
    def cmd_gui(_: argparse.Namespace) -> None:
        run_gui()

    p_gui = subparsers.add_parser("gui", help="Launch desktop GUI for pasting text and predicting")
    p_gui.set_defaults(func=cmd_gui)

    # IoT MQTT service
    def cmd_iot(args: argparse.Namespace) -> None:
        run_mqtt_service(
            broker=args.broker,
            port=args.port,
            sub_image_topic=args.sub_image_topic,
            sub_text_topic=args.sub_text_topic,
            pub_result_topic=args.pub_result_topic,
            model_path=Path(args.model_path),
            username=args.username,
            password=args.password,
            tesseract_cmd=args.tesseract_cmd,
        )

    p_iot = subparsers.add_parser("iot", help="Run MQTT service to receive text/images and publish predictions")
    p_iot.add_argument("--broker", type=str, default="localhost", help="MQTT broker host")
    p_iot.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    p_iot.add_argument("--username", type=str, default=None, help="MQTT username")
    p_iot.add_argument("--password", type=str, default=None, help="MQTT password")
    p_iot.add_argument("--sub-image-topic", type=str, default="esp32/essay_image", help="Subscribe topic for images")
    p_iot.add_argument("--sub-text-topic", type=str, default="esp32/essay_text", help="Subscribe topic for text")
    p_iot.add_argument("--pub-result-topic", type=str, default="esp32/essay_result", help="Publish topic for predictions")
    p_iot.add_argument("--model-path", type=str, default="models/model.joblib", help="Path to saved model")
    p_iot.add_argument("--tesseract-cmd", type=str, default=None, help="Path to tesseract executable if not on PATH")
    p_iot.set_defaults(func=cmd_iot)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

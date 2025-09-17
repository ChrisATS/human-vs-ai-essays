from __future__ import annotations

from pathlib import Path
from typing import List

import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from .io import ensure_dir


def gen_classification_report_text(y_true, y_pred, target_names: List[str]) -> str:
    return classification_report(y_true, y_pred, target_names=target_names, digits=4)


def gen_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def save_classification_report(report_text: str, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    out_path.write_text(report_text, encoding="utf-8")


def save_metrics_json(metrics: dict, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def plot_confusion_matrix(cm, labels: List[str], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(labels)), yticks=range(len(labels)),
        xticklabels=labels, yticklabels=labels,
        ylabel="True label", xlabel="Predicted label",
        title="Confusion Matrix",
    )
    # Rotate tick labels and set alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells
    thresh = cm.max() / 2.0 if cm.size else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Optional

from src.models.model import load_model


class PredictorGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Human vs AI Essay Classifier")
        self.root.geometry("820x600")
        self.root.minsize(700, 500)

        self.model_path_var = tk.StringVar(value=str(Path("models/model.joblib")))
        self.status_var = tk.StringVar(value="Model not loaded")
        self.pred_var = tk.StringVar(value="")
        self.score_var = tk.StringVar(value="")

        self._model = None
        self._loaded_path: Optional[Path] = None

        self._build_ui()

    def _build_ui(self) -> None:
        # Top: model path + load
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Model file:").pack(side=tk.LEFT)
        entry = ttk.Entry(top, textvariable=self.model_path_var, width=70)
        entry.pack(side=tk.LEFT, padx=6, fill=tk.X, expand=True)

        browse_btn = ttk.Button(top, text="Browse", command=self.on_browse)
        browse_btn.pack(side=tk.LEFT, padx=4)

        load_btn = ttk.Button(top, text="Load", command=self.on_load)
        load_btn.pack(side=tk.LEFT)

        status = ttk.Label(self.root, textvariable=self.status_var, padding=(10, 0))
        status.pack(anchor=tk.W)

        # Middle: text input
        mid = ttk.Frame(self.root, padding=10)
        mid.pack(fill=tk.BOTH, expand=True)

        ttk.Label(mid, text="Paste essay text below:").pack(anchor=tk.W)

        text_frame = ttk.Frame(mid)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 8))

        self.text_widget = tk.Text(text_frame, wrap=tk.WORD, height=18)
        self.text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scroll = ttk.Scrollbar(text_frame, command=self.text_widget.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_widget.configure(yscrollcommand=scroll.set)

        # Bottom: actions + result
        actions = ttk.Frame(self.root, padding=10)
        actions.pack(fill=tk.X)

        predict_btn = ttk.Button(actions, text="Predict", command=self.on_predict)
        predict_btn.pack(side=tk.LEFT)

        clear_btn = ttk.Button(actions, text="Clear", command=self.on_clear)
        clear_btn.pack(side=tk.LEFT, padx=6)

        result = ttk.Frame(self.root, padding=10)
        result.pack(fill=tk.X)

        ttk.Label(result, text="Prediction:").pack(side=tk.LEFT)
        pred_lbl = ttk.Label(result, textvariable=self.pred_var, font=("Segoe UI", 11, "bold"))
        pred_lbl.pack(side=tk.LEFT, padx=(6, 18))

        ttk.Label(result, text="Score:").pack(side=tk.LEFT)
        ttk.Label(result, textvariable=self.score_var).pack(side=tk.LEFT, padx=6)

    def on_browse(self) -> None:
        path = filedialog.askopenfilename(
            title="Select model file",
            filetypes=[("Joblib files", "*.joblib"), ("All files", "*.*")],
            initialdir=str(Path.cwd() / "models"),
        )
        if path:
            self.model_path_var.set(path)

    def on_load(self) -> None:
        path = Path(self.model_path_var.get()).expanduser()
        try:
            self._model = load_model(path)
            self._loaded_path = path
            self.status_var.set(f"Loaded: {path}")
        except Exception as e:
            self._model = None
            self._loaded_path = None
            messagebox.showerror("Load Error", f"Failed to load model:\n{e}")
            self.status_var.set("Model not loaded")

    def _ensure_model_loaded(self) -> bool:
        # Load if not loaded or path changed
        desired = Path(self.model_path_var.get()).expanduser()
        if self._model is None or self._loaded_path != desired:
            try:
                self._model = load_model(desired)
                self._loaded_path = desired
                self.status_var.set(f"Loaded: {desired}")
            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load model:\n{e}")
                self.status_var.set("Model not loaded")
                return False
        return True

    def on_predict(self) -> None:
        if not self._ensure_model_loaded():
            return
        text = self.text_widget.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Empty Text", "Please paste or type some text")
            return
        try:
            pred = int(self._model.predict([text])[0])
            label = "AI" if pred == 1 else "Human"
            self.pred_var.set(label)
            # Score via decision function if available
            score = None
            try:
                score = float(self._model.decision_function([text])[0])
            except Exception:
                score = None
            self.score_var.set(f"{score:.4f}" if score is not None else "N/A")
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Failed to predict:\n{e}")

    def on_clear(self) -> None:
        self.text_widget.delete("1.0", tk.END)
        self.pred_var.set("")
        self.score_var.set("")


def run_gui() -> None:
    root = tk.Tk()
    # Try to use themed widgets on Windows
    try:
        root.call("tk", "scaling", 1.25)
    except Exception:
        pass
    PredictorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()

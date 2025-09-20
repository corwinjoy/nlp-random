"""
Training script for fine-tuning DistilBERT on the
Hugging Face Emotion dataset with deterministic behavior where possible.

- Loads and tokenizes the dataset
- Fine-tunes a sequence classification head
- Evaluates on the validation split
- Plots a normalized confusion matrix

Adapted from: https://github.com/nlp-with-transformers/notebooks/blob/main/02_classification.ipynb
"""

from __future__ import annotations

# Standard library imports
import random

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)

import transformers
from datasets import load_dataset


def set_seeds(seed: int, deterministic: bool = False):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)

    # Try to enforce determinism
    if deterministic and torch.cuda.is_available():
        import os
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False  # Disable TF32 (which uses mixed precision)
        torch.backends.cudnn.allow_tf32 = False


def tokenize_function(tokenizer, batch):
    """Tokenize a batch of examples from the Emotion dataset."""
    # 'text' is the input field in the emotion dataset
    return tokenizer(batch["text"], padding=True, truncation=True)


def compute_metrics(pred):
    """Compute accuracy and weighted F1 from Trainer predictions."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }


def plot_confusion_matrix(y_preds, y_true, labels):
    """Plot a normalized confusion matrix for predictions vs. true labels."""
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.tight_layout()
    plt.show()


def main() -> None:
    # Configuration
    # MODEL_CKPT = "distilbert-base-uncased"
    MODEL_CKPT = "answerdotai/ModernBERT-base"
    BATCH_SIZE = 64
    NUM_EPOCHS = 2
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    SEED = 42  # Set seeds for deterministic-ish training

    # Setup
    set_seeds(SEED, True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and dataset
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_CKPT)
    emotions = load_dataset("emotion")

    # Tokenize the dataset. The label column is preserved by default.
    emotions_encoded = emotions.map(
        lambda batch: tokenize_function(tokenizer, batch),
        batched=True,
        batch_size=None,
    )

    # Restrict columns and set format to Torch tensors for Trainer
    emotions_encoded.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )

    # Build model for sequence classification
    num_labels = emotions["train"].features["label"].num_classes

    if False:
        # Generic transformer
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            MODEL_CKPT, num_labels=num_labels
        ).to(device)
    else:
        # ModernBERT transformer
        model = transformers.ModernBertForSequenceClassification.from_pretrained(
            MODEL_CKPT,
            num_labels=num_labels).to(device)

    # Training configuration
    logging_steps = max(1, len(emotions_encoded["train"]) // BATCH_SIZE)
    model_name = f"{MODEL_CKPT}-finetuned-emotion"
    training_args = transformers.TrainingArguments(
        output_dir=model_name,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",  # run eval at the end of each epoch
        disable_tqdm=False,
        logging_steps=logging_steps,
        push_to_hub=False,
        log_level="error",
        seed=SEED,
        data_seed=SEED,
    )

    # Trainer handles training loop, evaluation, and metrics
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=emotions_encoded["train"],
        eval_dataset=emotions_encoded["validation"],
        tokenizer=tokenizer,
    )

    # Train and evaluate
    trainer.train()

    preds_output = trainer.predict(emotions_encoded["validation"])
    print("Validation metrics:", preds_output.metrics)

    # Confusion matrix using predictions vs. true labels returned by Trainer
    y_preds = np.argmax(preds_output.predictions, axis=1)
    y_true = preds_output.label_ids
    label_names = emotions["train"].features["label"].names
    plot_confusion_matrix(y_preds, y_true, label_names)


if __name__ == "__main__":
    main()

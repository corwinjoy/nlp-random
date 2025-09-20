# Adapted from https://github.com/nlp-with-transformers/notebooks/blob/main/02_classification.ipynb

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import Trainer, TrainingArguments
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

emotions = load_dataset("emotion")
emotions.set_format(type="pandas")
df = emotions["train"][:]
df.head()


model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

#from transformers import DistilBertTokenizer
#distilbert_tokenizer = DistilBertTokenizer.from_pretrained(model_ckpt)

def tokenize(batch):
    return tokenizer(batch["text"].to_list(), padding=True, truncation=True)

emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
for split in ("train", "test", "validation"):
    emotions_encoded[split] = emotions_encoded[split].add_column(name="label", column=emotions[split]["label"])

print(emotions_encoded["train"].column_names)
emotions_encoded.set_format("torch",
                            columns=["input_ids", "attention_mask", "label"])


base_model = AutoModel.from_pretrained(model_ckpt).to(device)

def extract_hidden_states(batch):
    # Place model inputs on the GPU
    inputs = {k:v.to(device) for k,v in batch.items()
              if k in tokenizer.model_input_names}
    # Extract last hidden states
    with torch.no_grad():
        last_hidden_state = base_model(**inputs).last_hidden_state
    # Return vector for [CLS] token
    return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}

emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True)

X_train = np.array(emotions_hidden["train"]["hidden_state"])
X_valid = np.array(emotions_hidden["validation"]["hidden_state"])
y_train = np.array(emotions_hidden["train"]["label"])
y_valid = np.array(emotions_hidden["validation"]["label"])
labels = emotions["train"].features["label"].names


from transformers import AutoModelForSequenceClassification

num_labels = 6
seq_model = (AutoModelForSequenceClassification
         .from_pretrained(model_ckpt, num_labels=num_labels)
         .to(device))

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

# May need to login
# huggingface-cli login

batch_size = 64
logging_steps = len(emotions_encoded["train"]) // batch_size
model_name = f"{model_ckpt}-finetuned-emotion"
training_args = TrainingArguments(output_dir=model_name,
                                  num_train_epochs=2,
                                  learning_rate=2e-5,
                                  per_device_train_batch_size=batch_size,
                                  per_device_eval_batch_size=batch_size,
                                  weight_decay=0.01,
                                  eval_strategy="epoch",
                                  disable_tqdm=False,
                                  logging_steps=logging_steps,
                                  push_to_hub=False,
                                  log_level="error")


trainer = Trainer(model=seq_model, args=training_args,
                  compute_metrics=compute_metrics,
                  train_dataset=emotions_encoded["train"],
                  eval_dataset=emotions_encoded["validation"],
                  tokenizer=tokenizer)
trainer.train()

preds_output = trainer.predict(emotions_encoded["validation"])
print(preds_output.metrics)

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()
y_preds = np.argmax(preds_output.predictions, axis=1)

plot_confusion_matrix(y_preds, y_valid, labels)



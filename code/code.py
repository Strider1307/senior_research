# %%
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================
# 1. LOAD FIRST 100 ROWS OF DATASET
# ============================================================
df = pd.read_csv("phishing_email 1000.csv", nrows=1000)

texts = df["text_combined"].tolist()
labels = df["label"].tolist()  # 0 = legitimate, 1 = phishing

# ============================================================
# 2. FUNCTION: RUN MODEL INFERENCE
# ============================================================
def run_model(model_name, texts):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    preds = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=1).item()
        preds.append(pred)

    return preds

# ============================================================
# 3. FUNCTION: EVALUATE MODEL
# ============================================================
def evaluate(model_name, texts, labels):
    preds = run_model(model_name, texts)

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    cm = confusion_matrix(labels, preds)

    return {
        "preds": preds,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cm": cm
    }

# ============================================================
# 4. RUN BOTH MODELS
# ============================================================
results_base = evaluate("google-bert/bert-base-uncased", texts, labels)
results_finetuned = evaluate("ealvaradob/bert-finetuned-phishing", texts, labels)

# ============================================================
# 5. PRINT METRICS TABLE
# ============================================================
print("\n=== METRICS COMPARISON ===")
print(f"BERT-Base: Acc={results_base['accuracy']:.3f}, "
      f"Prec={results_base['precision']:.3f}, "
      f"Rec={results_base['recall']:.3f}, "
      f"F1={results_base['f1']:.3f}")

print(f"Fine-Tuned: Acc={results_finetuned['accuracy']:.3f}, "
      f"Prec={results_finetuned['precision']:.3f}, "
      f"Rec={results_finetuned['recall']:.3f}, "
      f"F1={results_finetuned['f1']:.3f}")

# ============================================================
# 6. FUNCTION: PLOT & SAVE CONFUSION MATRIX
# ============================================================
def plot_confusion_matrix(cm, title, filename):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legitimate", "Phishing"],
                yticklabels=["Legitimate", "Phishing"])
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# ============================================================
# 7. GENERATE BOTH CONFUSION MATRICES
# ============================================================
plot_confusion_matrix(results_base["cm"],
                      "Confusion Matrix — BERT Base",
                      "cm_baseline.png")

plot_confusion_matrix(results_finetuned["cm"],
                      "Confusion Matrix — Fine-Tuned BERT",
                      "cm_finetuned.png")

print("\nSaved confusion matrices as:")
print(" - cm_baseline.png")
print(" - cm_finetuned.png")

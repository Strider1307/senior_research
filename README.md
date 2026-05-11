# An Analysis of Social Engineering Attacks on AI Detection Model Security

This project investigates how phishing email detection systems do against adversarial attacks, particularly those targeting transformer-based models like BERT and large language models such as ChatGPT.
##Introduction:
  Phishing remains one of the most persistent forms of social engineering because attackers
use deceptive email content to manipulate users into revealing credentials, downloading
malware, or trusting fraudulent communications. Recent security writing also suggests that
generative AI is making phishing campaigns more scalable and more convincing by helping
attackers produce natural-sounding content at higher volume [1]. As a result, phishing email
detection has become an important natural language processing problem in which
contextual understanding is more useful than simple keyword matching [2]. ​
  Transformer-based models such as BERT are well suited to text classification because they
encode bidirectional context rather than relying only on local word patterns [3]. The bert-
base-uncased model released by Google is a general-purpose uncased English BERT
model with 110 million parameters that was pre-trained using masked language modeling
and is widely used as a baseline for downstream NLP tasks [3]. In contrast, bert-finetuned-
phishing is a task-specific phishing model built on bert-large-uncased; its Hugging Face
model card states that it was fine-tuned for phishing detection across URLs, emails, SMS
messages, and websites, and reports evaluation metrics including 0.9717 accuracy, 0.9658
precision, and 0.9670 recall [4]. These two models provide a meaningful comparison
between a general pretrained BERT baseline and a phishing-specialized fine-tuned model
[3][4]. ​
  Prior literature supports the use of BERT-like architectures for phishing-related classification
tasks. Dahl and Gustafsson found that BERT slightly outperformed manual text pattern
matching in automated email text classification and achieved a statistically significant
improvement in recall, suggesting that contextual models better handle linguistic variation in
email messages [5]. Songailaitė and colleagues reported that fine-tuned BERT-based
models for phishing detection in emails achieved performance above 0.985 across
accuracy, precision, and recall, showing that transformer fine-tuning can be highly effective
for phishing classification [2]. Zhao likewise describes BERT as a strong approach for
phishing email detection because it captures word context more effectively than simpler
methods and can be adapted to security tasks involving text-based phishing content . ​
  The objective of this paper is to compare a general-purpose BERT model, bert-base-
uncased, with a phishing-specific fine-tuned BERT model., It uses current research and
model documentation to explain why the bert-finetuned-phishing model will perform better,
how the models will be evaluated, and why this comparison matters for phishing detection
systems​

##Methods:
###Dataset:​
A labeled phishing‑email dataset (phishing_email.csv) was used. The first 100 samples were selected for evaluation. Each entry contained raw email text and a binary label (0 = legitimate, 1 = phishing). Text and labels were extracted using pandas.​

###Models​:
Two transformer models were evaluated:​
BERT‑Base Uncased — general‑purpose pretrained model. A new classification head was automatically initialized, producing expected “missing/unexpected key” warnings. Phishing‑Fine‑Tuned BERT — pretrained and fine‑tuned specifically for phishing detection, loading cleanly with a compatible classification head. Both models were loaded using Hugging Face AutoTokenizer and AutoModelForSequenceClassification.​

###Preprocessing:​
Emails were tokenized using WordPiece with truncation and dynamic padding. No additional text cleaning was applied to preserve phishing‑specific linguistic cues.​

###Evaluation:​
Both models processed the same 100‑email subset. Predictions were generated via argmax over output logits. Performance was assessed using: Accuracy, Precision, Recall, F1 Score, and a Confusion Matrix​
Metrics were computed with scikit‑learn. Confusion matrices were visualized using Seaborn/Matplotlib and saved as PNG files for reproducibility in VS Code.​

###Environment​

Experiments were run in Python using:​

transformers, torch, pandas, numpy, scikit‑learn, matplotlib, and seaborn.​

##Summary:
The project involves comparing the results of a general BERT model against those obtained from a phishing-focused BERT model in order to check whether fine-tuning on specific tasks increases model performance. In particular, phishing is viewed as a critical threat caused by social engineering activities. The draft notes that transformers are effective tools in detecting the problem since they analyze the meaning in the context, and not the occurrence of particular keywords. The paper suggests that the general BERT model, namely bert-base-uncased, will act as a baseline, while bert-finetuned-phishing would be used to detect phishing emails, thus being more accurate. The methods part claims that the models should work on the same data set and produce accuracy, precision, recall, and F1 scores; however, there is still the need for more specific information concerning the data set, Python packages used, and metric calculation procedures.

##Citations:
[1] Inanna, “Generative AI and Phishing: The New Face of Social Engineering,” DataSunrise. Accessed: Feb. 25, 2026. [Online]. Available: https://www.datasunrise.com/knowledge-center/ai-security/generative-ai-and-phishing/ ​

[2] M. Songailaitė, E. Kankevičiūtė, B. Zhyhun, and J. Mandravickaitė, “BERT-Based Models for Phishing Detection”. ​

[3] “google-bert/bert-base-uncased · Hugging Face.” Accessed: Apr. 20, 2026. [Online]. Available: https://huggingface.co/google-bert/bert-base-uncased ​

[4] “ealvaradob/bert-finetuned-phishing · Hugging Face.” Accessed: Apr. 20, 2026. [Online]. Available: https://huggingface.co/ealvaradob/bert-finetuned-phishing ​

[5] S. Dahl and R. Gustafsson, “Evaluating BERT for Email Text Classification:”. ​

[6] H. Zhao, “Deep Learning in Security: Text-based Phishing Email Detection with BERT Model,” Splunk. Accessed: Feb. 25, 2026. [Online]. Available: https://www.splunk.com/en_us/blog/security/deep-learning-in-security-text-based-phishing-email-detection-with-bert-model.html 


##Models:
BERT base model (uncased): https://huggingface.co/google-bert/bert-base-uncased

BERT FINETUNED ON PHISHING DETECTION: https://huggingface.co/ealvaradob/bert-finetuned-phishing


##Code Used:

'

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

df = pd.read_csv("phishing_email 1000.csv", nrows=1000)

texts = df["text_combined"].tolist()
labels = df["label"].tolist()  # 0 = legitimate, 1 = phishing

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
results_base = evaluate("google-bert/bert-base-uncased", texts, labels)
results_finetuned = evaluate("ealvaradob/bert-finetuned-phishing", texts, labels)

print("\n=== METRICS COMPARISON ===")
print(f"BERT-Base: Acc={results_base['accuracy']:.3f}, "
      f"Prec={results_base['precision']:.3f}, "
      f"Rec={results_base['recall']:.3f}, "
      f"F1={results_base['f1']:.3f}")

print(f"Fine-Tuned: Acc={results_finetuned['accuracy']:.3f}, "
      f"Prec={results_finetuned['precision']:.3f}, "
      f"Rec={results_finetuned['recall']:.3f}, "
      f"F1={results_finetuned['f1']:.3f}")

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

plot_confusion_matrix(results_base["cm"],
                      "Confusion Matrix — BERT Base",
                      "cm_baseline.png")

plot_confusion_matrix(results_finetuned["cm"],
                      "Confusion Matrix — Fine-Tuned BERT",
                      "cm_finetuned.png")

print("\nSaved confusion matrices as:")
print(" - cm_baseline.png")
print(" - cm_finetuned.png")
'


# 🧠 BERT Fine-Tuning for Grammar Classification

This project demonstrates fine-tuning of a pre-trained BERT model to classify English sentences as **grammatically correct** or **incorrect**.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Details](#training-details)
- [Evaluation](#evaluation)
- [Interactive Interface](#interactive-interface)
- [Model Export](#model-export)
- [Usage](#usage)
- [Requirements](#requirements)
- [License](#license)

---

## 📖 Overview

This project focuses on binary classification using the BERT architecture:
- Input: An English sentence
- Output: `Grammatically Correct` or `Grammatically Incorrect`

Fine-tuning is performed using the [Transformers](https://huggingface.co/transformers/) library by Hugging Face, specifically using `BertForSequenceClassification`.

---

## 📁 Dataset

The dataset consists of tab-separated sentences labeled as grammatically **correct (1)** or **incorrect (0)**. It includes:
- Training set
- Validation set
- Holdout/Out-of-domain test set (`out_of_domain_dev.tsv`)

Each sample contains:
- Sentence source
- Label
- Sentence text

---

## 🏗️ Model Architecture

- **Base Model**: BERT-Base (uncased)
- **Tokenizer**: `bert-base-uncased`
- **Classification Head**: Fully connected layer on top of [CLS] token output

---

## ⚙️ Training Details

- **Optimizer**: `AdamW`
- **Learning Rate**: `2e-5`
- **Batch Size**: `32`
- **Max Sequence Length**: `128`
- **Epochs**: `4`
- **Loss Function**: CrossEntropy
- **Scheduler**: Linear warmup scheduler

---

## 📈 Evaluation

Metrics used:
- **Validation Accuracy**
- **Matthews Correlation Coefficient (MCC)**

The evaluation is done on both validation and holdout datasets. Sample output includes:
- Sentence
- Raw logits
- Softmax probabilities
- Predicted class
- True label

---

## 💡 Interactive Interface

An interactive widget-based UI is implemented using `ipywidgets`:

```python
import ipywidgets as widgets
```
Users can type a sentence
BERT predicts and displays whether the sentence is grammatically correct or not

## 💾 Model Export
The trained model and tokenizer are saved locally and can be zipped for easy sharing or reuse:
```python
model.save_pretrained('/content/model')
tokenizer.save_pretrained('/content/model')
```
Zipping:
```python
shutil.make_archive('/content/drive/MyDrive/ANPR/BERT Fine-Tuning Sentence Classification', 'zip', '/content/model')
```
## 🚀 Usage
After saving the model, you can reload it in a new session like this:
```python
from transformers import BertTokenizer, BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained('/content/model/')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
```
To predict:
```python
def predict(sentence, model, tokenizer):
    ...
    return predicted_label
```

## 🧰 Requirements
```bash
pip install transformers
pip install torch
pip install scikit-learn
pip install ipywidgets
```
Make sure you also enable widgets in Jupyter/Colab:
```bash
jupyter nbextension enable --py widgetsnbextension --sys-prefix
```

## 📜 License
This project is open-source and licensed under the MIT License.

## 🙌 Acknowledgements
- **HuggingFace Transformers**

- **BERT: Pre-training of Deep Bidirectional Transformers**

- **Special thanks to the dataset providers.**



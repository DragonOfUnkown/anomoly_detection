# Network Anomaly Detection using Random Forest

This project demonstrates the development of a multi-class network intrusion detection model using the NSL-KDD dataset and the Random Forest algorithm from `scikit-learn`.

## 📦 Features

- Multi-class classification for network attack types
- Preprocessing including:
  - Binary and multi-class label mapping
  - One-hot encoding of categorical features
  - Selection of critical numeric features
- Model evaluation with accuracy, precision, recall, F1-score
- Visualization using confusion matrices
- Model saving and optional upload to a local API

## 📂 Dataset

- **Source:** [NSL-KDD Dataset on HackTheBox Academy](https://academy.hackthebox.com/storage/modules/292/KDD_dataset.zip)
- **File used:** `KDD+.txt`
- The dataset includes 41 features and labels indicating the type of traffic (normal or attack).

## 🔧 Installation

pip install pandas numpy scikit-learn seaborn matplotlib joblib

## 🧠 Model Training

- Uses `RandomForestClassifier` from `scikit-learn`
- Splits data into:
  - 80% training/test split
  - 20% reserved for test set
  - 30% of the training portion is used for validation
- Features include:
  - Protocol type
  - Service type
  - Duration
  - Source/Destination bytes
  - Various statistical and session-based metrics

---

## 🎯 Labels

### Binary (`attack_flag`)
- `0`: normal
- `1`: any attack

### Multi-Class (`attack_map`)
- `0`: normal
- `1`: DoS (Denial of Service)
- `2`: Probe (scanning)
- `3`: Privilege Escalation
- `4`: Unauthorized Access

---

## 📈 Evaluation

Model performance is evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix (visualized using `seaborn`)

Validation and test results are printed and displayed with seaborn heatmaps for clarity.

---

## 💾 Save & Export

The trained model is saved to a `.joblib` file:
```bash
network_anomaly_detection_model.joblib

# Importing necessary libraries
import requests, zipfile, io, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import json

# Download and extract NSL-KDD dataset
url = "https://academy.hackthebox.com/storage/modules/292/KDD_dataset.zip"
response = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(response.content))
z.extractall('.')

# Define column names and read data
file_path = 'KDD+.txt'
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 
    'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack', 'level'
]
df = pd.read_csv(file_path, names=columns)

# Binary target
df['attack_flag'] = df['attack'].apply(lambda a: 0 if a == 'normal' else 1)

# Multi-class target mapping
dos_attacks = ['apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod', 
               'processtable', 'smurf', 'teardrop', 'udpstorm', 'worm']
probe_attacks = ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']
privilege_attacks = ['buffer_overflow', 'loadmdoule', 'perl', 'ps', 
                     'rootkit', 'sqlattack', 'xterm']
access_attacks = ['ftp_write', 'guess_passwd', 'http_tunnel', 'imap', 
                  'multihop', 'named', 'phf', 'sendmail', 'snmpgetattack', 
                  'snmpguess', 'spy', 'warezclient', 'warezmaster', 
                  'xclock', 'xsnoop']

def map_attack(attack):
    if attack in dos_attacks: return 1
    elif attack in probe_attacks: return 2
    elif attack in privilege_attacks: return 3
    elif attack in access_attacks: return 4
    else: return 0
df['attack_map'] = df['attack'].apply(map_attack)

# One-hot encode categorical features
encoded = pd.get_dummies(df[['protocol_type', 'service']])

# Select numeric features
numeric_features = [
    'duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot', 
    'num_failed_logins', 'num_compromised', 'root_shell', 'su_attempted', 
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 
    'num_outbound_cmds', 'count', 'srv_count', 'serror_rate', 
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 
    'dst_host_srv_rerror_rate'
]
train_set = encoded.join(df[numeric_features])
multi_y = df['attack_map']

# Split dataset
train_X, test_X, train_y, test_y = train_test_split(train_set, multi_y, test_size=0.2, random_state=1337)
multi_train_X, multi_val_X, multi_train_y, multi_val_y = train_test_split(train_X, train_y, test_size=0.3, random_state=1337)

# Train model
rf_model_multi = RandomForestClassifier(random_state=1337)
rf_model_multi.fit(multi_train_X, multi_train_y)

# Validation evaluation
multi_predictions = rf_model_multi.predict(multi_val_X)
print("Validation Set Evaluation:")
print(f"Accuracy: {accuracy_score(multi_val_y, multi_predictions):.4f}")
print(f"Precision: {precision_score(multi_val_y, multi_predictions, average='weighted'):.4f}")
print(f"Recall: {recall_score(multi_val_y, multi_predictions, average='weighted'):.4f}")
print(f"F1-Score: {f1_score(multi_val_y, multi_predictions, average='weighted'):.4f}")
sns.heatmap(confusion_matrix(multi_val_y, multi_predictions), annot=True, fmt='d', cmap='Blues')
plt.title('Validation Set Confusion Matrix')
plt.show()
print(classification_report(multi_val_y, multi_predictions))

# Test evaluation
test_multi_predictions = rf_model_multi.predict(test_X)
print("Test Set Evaluation:")
print(f"Accuracy: {accuracy_score(test_y, test_multi_predictions):.4f}")
print(f"Precision: {precision_score(test_y, test_multi_predictions, average='weighted'):.4f}")
print(f"Recall: {recall_score(test_y, test_multi_predictions, average='weighted'):.4f}")
print(f"F1-Score: {f1_score(test_y, test_multi_predictions, average='weighted'):.4f}")
sns.heatmap(confusion_matrix(test_y, test_multi_predictions), annot=True, fmt='d', cmap='Blues')
plt.title('Test Set Confusion Matrix')
plt.show()
print(classification_report(test_y, test_multi_predictions))

# Save model
model_filename = 'network_anomaly_detection_model.joblib'
joblib.dump(rf_model_multi, model_filename)
print(f"Model saved to {model_filename}")



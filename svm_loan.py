# ================================
# LOAN PREDICTION WITH SVM (FULL CODE)
# ================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler

import warnings
warnings.filterwarnings('ignore')

print("Loading dataset...")

# ================================
# LOAD DATASET
# ================================
try:
    df = pd.read_csv('loan_data (2).csv')
except:
    df = pd.read_csv('loan_data.csv')

print(df.head())
print(df.info())
print(df.describe())

# ================================
# EDA — PIE CHART
# ================================
temp = df['Loan_Status'].value_counts()
plt.pie(temp.values, labels=temp.index, autopct='%1.1f%%')
plt.title("Loan Status Distribution")
plt.show()

# ================================
# COUNT PLOT — GENDER & MARRIED
# ================================
plt.subplots(figsize=(15, 5))
for i, col in enumerate(['Gender', 'Married']):
    plt.subplot(1, 2, i+1)
    sb.countplot(data=df, x=col, hue='Loan_Status')
plt.tight_layout()
plt.show()

# ================================
# DISTPLOT — INCOME & LOAN AMOUNT
# ================================
plt.subplots(figsize=(15, 5))
for i, col in enumerate(['ApplicantIncome', 'LoanAmount']):
    plt.subplot(1, 2, i+1)
    sb.distplot(df[col])
plt.tight_layout()
plt.show()

# ================================
# BOX PLOT (OUTLIERS)
# ================================
plt.subplots(figsize=(15, 5))
for i, col in enumerate(['ApplicantIncome', 'LoanAmount']):
    plt.subplot(1, 2, i+1)
    sb.boxplot(df[col])
plt.tight_layout()
plt.show()

# REMOVE EXTREME OUTLIERS
df = df[df['ApplicantIncome'] < 25000]
df = df[df['LoanAmount'] < 400000]

# ================================
# LABEL ENCODING
# ================================
def encode_labels(data):
    for col in data.columns:
        if data[col].dtype == 'object':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
    return data

df = encode_labels(df)

# ================================
# HEATMAP CORRELATION
# ================================
plt.figure(figsize=(10, 8))
sb.heatmap(df.corr() > 0.8, annot=True, cbar=False)
plt.title("High Correlation Heatmap")
plt.show()

# ================================
# TRAIN / TEST SPLIT
# ================================
features = df.drop('Loan_Status', axis=1)
target = df['Loan_Status'].values

X_train, X_val, Y_train, Y_val = train_test_split(
    features, target,
    test_size=0.2,
    random_state=10
)

# ================================
# HANDLE IMBALANCED DATA
# ================================
ros = RandomOverSampler(sampling_strategy='minority', random_state=0)
X, Y = ros.fit_resample(X_train, Y_train)

# ================================
# NORMALIZATION
# ================================
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_val = scaler.transform(X_val)

# ================================
# TRAIN SVM MODEL
# ================================
model = SVC(kernel='rbf')
model.fit(X, Y)

print("\n===== RESULTS =====")
print("Training ROC AUC:", roc_auc_score(Y, model.predict(X)))
print("Validation ROC AUC:", roc_auc_score(Y_val, model.predict(X_val)))

# ================================
# CONFUSION MATRIX
# ================================
cm = confusion_matrix(Y_val, model.predict(X_val))

plt.figure(figsize=(6, 6))
sb.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ================================
# CLASSIFICATION REPORT
# ================================
print("\nClassification Report:")
print(classification_report(Y_val, model.predict(X_val)))

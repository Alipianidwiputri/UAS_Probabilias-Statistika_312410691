# UAS_Probabilias-Statistika

# Loan Eligibility Prediction using Machine Learning

**Project UAS - Probabilitas dan Statistika**

**Prediksi kelayakan pinjaman menggunakan algoritma Support Vector Machine (SVM) dengan Python**

**Nama: Alipiani Dwi Putri**

**NIM: 312410691**

**Kelas: TI 24 A2**

**Mata Kuliah: Probabilitas dan Statistika**

**Dosen Pengampu: Dr. Muhamad Fatchan, S.Kom., M.Kom**

---

**Link Video Penjelasan**


---

##  Deskripsi Project

Project ini menggunakan Machine Learning untuk memprediksi apakah seseorang layak mendapatkan pinjaman atau tidak berdasarkan data historis pemohon. Algoritma yang digunakan adalah **Support Vector Machine (SVM)** dengan kernel RBF yang efektif untuk klasifikasi binary dan dapat menangani data non-linear.

**Tujuan:**
- Mengotomatisasi proses prediksi kelayakan pinjaman
- Meningkatkan akurasi pengambilan keputusan
- Mengurangi risiko kredit macet
- Mempercepat proses approval pinjaman

---

## Teknologi yang Digunakan

### Libraries Python:
- **NumPy**: Komputasi numerik dan operasi array
- **Pandas**: Manipulasi dan analisis data
- **Matplotlib**: Visualisasi data dasar
- **Seaborn**: Visualisasi statistik yang estetik
- **Scikit-learn**: Machine Learning algorithms dan tools
- **Imbalanced-learn**: Handling imbalanced datasets

### Tools:
- Python 3.x
- Jupyter Notebook / VS Code
- Git & GitHub

---

## Dataset

Dataset berisi informasi pemohon pinjaman dengan fitur-fitur berikut:

| Kolom | Deskripsi | Tipe Data |
|-------|-----------|-----------|
| Loan_ID | ID unik pemohon | Object |
| Gender | Jenis kelamin (Male/Female) | Object |
| Married | Status pernikahan (Yes/No) | Object |
| ApplicantIncome | Pendapatan pemohon | Integer |
| LoanAmount | Jumlah pinjaman yang dimohonkan | Float |
| Loan_Status | Status approval (Y/N) - Target | Object |

**Ukuran Dataset:** 598 baris × 6 kolom

---

##  Metodologi

### 1. Import Library

```python
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
```

**Penjelasan:**
- **NumPy & Pandas**: Untuk manipulasi data dan operasi matematika
- **Matplotlib & Seaborn**: Untuk visualisasi dan exploratory data analysis
- **Scikit-learn**: Menyediakan algoritma SVM, preprocessing tools, dan metrics evaluasi
- **Imbalanced-learn**: Menangani dataset dengan class imbalance menggunakan oversampling
- **warnings.filterwarnings**: Menyembunyikan warning messages agar output lebih bersih

---

### 2. Loading Dataset

```python
try:
    df = pd.read_csv('loan_data (2).csv')
except:
    df = pd.read_csv('loan_data.csv')

print(df.head())
print(df.info())
print(df.describe())
```

**Penjelasan:**
- **pd.read_csv()**: Membaca file CSV dan convert menjadi DataFrame
- **df.head()**: Menampilkan 5 baris pertama untuk melihat struktur data
- **df.info()**: Menampilkan informasi detail (tipe data, missing values, memory usage)
- **df.describe()**: Statistik deskriptif (mean, std, min, max, quartiles)

**Output Penting:**
- Dataset memiliki 598 entries
- Kolom LoanAmount memiliki 21 missing values (577 non-null dari 598)
- ApplicantIncome berkisar dari 150 hingga 81,000
- LoanAmount berkisar dari 9 hingga 650

---

## Exploratory Data Analysis (EDA)

### 3.1 Pie Chart - Distribusi Loan Status

```python
temp = df['Loan_Status'].value_counts()
plt.pie(temp.values, labels=temp.index, autopct='%1.1f%%')
plt.title("Loan Status Distribution")
plt.show()
```

**Penjelasan:**
- **value_counts()**: Menghitung frekuensi setiap unique value
- **autopct='%1.1f%%'**: Menampilkan persentase dengan 1 desimal

**Output**




<img width="928" height="696" alt="image" src="https://github.com/user-attachments/assets/192af9e2-d748-4847-8821-d936f1704e34" />








- Menunjukkan proporsi antara pinjaman yang disetujui (Y) vs ditolak (N)
- Membantu mengidentifikasi **class imbalance** dalam dataset
- Jika distribusi tidak seimbang (misal 70%-30%), perlu teknik oversampling/undersampling
- Class imbalance bisa menyebabkan model bias terhadap kelas mayoritas

---

### 3.2 Count Plot - Gender & Married Status

```python
plt.subplots(figsize=(15, 5))
for i, col in enumerate(['Gender', 'Married']):
    plt.subplot(1, 2, i+1)
    sb.countplot(data=df, x=col, hue='Loan_Status')
plt.tight_layout()
plt.show()
```

**Penjelasan:**
- **subplots(figsize=(15, 5))**: Membuat figure dengan ukuran 15×5 inches
- **subplot(1, 2, i+1)**: Membuat 1 baris, 2 kolom subplot
- **hue='Loan_Status'**: Membedakan warna berdasarkan status pinjaman

**Output**





<img width="940" height="367" alt="image" src="https://github.com/user-attachments/assets/44aebf77-9f95-4c77-b291-5f9d9ad5ea59" />







- **Gender Plot**: Melihat apakah ada bias gender dalam approval pinjaman
- **Married Plot**: Menunjukkan pengaruh status pernikahan terhadap approval
- Dari analisis terlihat bahwa **married applicants** cenderung mengajukan pinjaman lebih besar
- Status pernikahan bisa menjadi faktor penting dalam keputusan approval

---

### 3.3 Distribution Plot - Income & Loan Amount

```python
plt.subplots(figsize=(15, 5))
for i, col in enumerate(['ApplicantIncome', 'LoanAmount']):
    plt.subplot(1, 2, i+1)
    sb.distplot(df[col])
plt.tight_layout()
plt.show()
```


**Penjelasan:**
- **distplot**: Menggabungkan histogram dengan KDE (Kernel Density Estimate) curve
- Visualisasi ini menampilkan distribusi continuous variables

**Output**





<img width="832" height="324" alt="image" src="https://github.com/user-attachments/assets/9520a518-2c8b-4e56-9618-f130f045745f" />






- **ApplicantIncome Distribution**: 
  - Menunjukkan **right-skewed distribution** (ekor panjang di kanan)
  - Mayoritas pemohon memiliki income di range rendah-menengah
  - Beberapa outliers dengan income sangat tinggi
- **LoanAmount Distribution**:
  - Juga menunjukkan right-skewed pattern
  - Sebagian besar pinjaman berada di range 100,000 - 200,000
  - Right-skewed adalah normal untuk financial data

---

### 3.4 Box Plot - Outlier Detection

```python
plt.subplots(figsize=(15, 5))
for i, col in enumerate(['ApplicantIncome', 'LoanAmount']):
    plt.subplot(1, 2, i+1)
    sb.boxplot(df[col])
plt.tight_layout()
plt.show()

# REMOVE EXTREME OUTLIERS
df = df[df['ApplicantIncome'] < 25000]
df = df[df['LoanAmount'] < 400000]
```

**Penjelasan:**
- **boxplot**: Menampilkan quartiles (Q1, median/Q2, Q3) dan outliers
- **Outliers**: Data points yang berada beyond 1.5 × IQR dari quartiles
- **IQR (Interquartile Range)**: Q3 - Q1

**Output**




<img width="847" height="330" alt="image" src="https://github.com/user-attachments/assets/5018ecbc-aa26-4b80-a602-648be74e0168" />







- **ApplicantIncome Boxplot**:
  - Banyak outliers di sisi atas (income sangat tinggi)
  - Median income sekitar 3,800
  - Outliers bisa dari: data entry errors, genuine extreme values, atau fraud
- **LoanAmount Boxplot**:
  - Outliers menunjukkan permintaan pinjaman yang sangat besar
  - Median loan amount sekitar 127,000

**Alasan Menghapus Outliers:**
- ApplicantIncome > 25,000 dihapus (extremely high income)
- LoanAmount > 400,000 dihapus (extremely large loans)
- Outliers ekstrem bisa distort model training
- Trade-off: lose some data (~2-3%) vs improve model stability

---

### 3.5 Label Encoding

```python
def encode_labels(data):
    for col in data.columns:
        if data[col].dtype == 'object':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
    return data

df = encode_labels(df)
```

**Penjelasan:**
- Machine Learning algorithms hanya bisa memproses data numerik
- **LabelEncoder** mengubah categorical variables menjadi integers
- Setiap unique category diberi integer unik (0, 1, 2, ...)

**Transformasi:**
- Gender: Male → 1, Female → 0
- Married: Yes → 1, No → 0
- Loan_Status: Y → 1, N → 0

**Catatan:**
- Label Encoding cocok untuk ordinal data atau tree-based models
- Alternatif: One-Hot Encoding untuk purely nominal variables

---

### 3.6 Correlation Heatmap

```python
plt.figure(figsize=(10, 8))
sb.heatmap(df.corr() > 0.8, annot=True, cbar=False)
plt.title("High Correlation Heatmap")
plt.show()
```

**Penjelasan:**
- **df.corr()**: Menghitung Pearson correlation coefficient antar numerical columns
- **> 0.8**: Filter hanya korelasi kuat (threshold 0.8)
- **annot=True**: Menampilkan nilai korelasi di dalam cells

**Output**




<img width="936" height="448" alt="image" src="https://github.com/user-attachments/assets/c7d3e3b7-5507-46ee-9b07-144b22ec8202" />








- **Korelasi berkisar dari -1 hingga +1**:
  - +1: Perfect positive correlation
  - 0: No correlation
  - -1: Perfect negative correlation
- **Multicollinearity**: High correlation antar predictor variables
  - Bisa membuat model coefficients unstable
  - Features redundant (memberikan info yang sama)
  - Solution: remove salah satu atau combine features

---

##  Data Preprocessing

### 4. Train-Test Split

```python
features = df.drop('Loan_Status', axis=1)
target = df['Loan_Status'].values

X_train, X_val, Y_train, Y_val = train_test_split(
    features, target,
    test_size=0.2,
    random_state=10
)
```

**Penjelasan:**
- **features (X)**: Semua kolom kecuali target variable
- **target (Y)**: Kolom yang akan diprediksi (Loan_Status)
- **test_size=0.2**: 20% data untuk validation, 80% untuk training
- **random_state=10**: Seed untuk reproducibility (hasil sama setiap run)

**Mengapa Split Data?**
- **Training set**: Untuk melatih model (learn patterns)
- **Validation set**: Untuk evaluate performa pada unseen data
- Mencegah **overfitting** (model hafal training data tapi gagal di new data)
- Rule of thumb: minimal 10× samples dari jumlah features

---

### 5. Handle Imbalanced Data

```python
ros = RandomOverSampler(sampling_strategy='minority', random_state=0)
X, Y = ros.fit_resample(X_train, Y_train)
```

**Penjelasan:**
- **Class Imbalance Problem**: Jika 70% approved dan 30% rejected
- Model bisa achieve 70% accuracy dengan selalu predict 'approved'
- **RandomOverSampler**: Duplicate samples dari minority class hingga balanced

**Cara Kerja:**
- Identifies minority class (kelas dengan samples lebih sedikit)
- Randomly duplicates samples dari minority class
- Hasil: balanced dataset dengan jumlah sama untuk kedua kelas

**Alternatif Teknik:**
1. **Undersampling**: Menghapus samples dari majority class
2. **SMOTE**: Membuat synthetic samples (bukan duplikasi)
3. **Class Weights**: Assign higher weight ke minority class

**Catatan Penting:**
- Oversampling hanya diterapkan pada **training set**
- **Validation set tidak di-oversample** (tetap original distribution)
- Tujuan: Agar evaluation realistic terhadap real-world condition

---

### 6. Normalization

```python
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_val = scaler.transform(X_val)
```

**Penjelasan:**
- **StandardScaler**: Mentransform features ke mean=0 dan std=1
- **Formula**: z = (x - μ) / σ
  - x = original value
  - μ = mean
  - σ = standard deviation

**Contoh Transformasi:**
```
Original Income: [3000, 5000, 8000]
Mean (μ): 5333
Std (σ): 2055
Scaled: [-1.14, -0.16, 1.30]
```

**Mengapa Scaling Penting untuk SVM?**
- SVM menggunakan **distance calculations** antar data points
- Features dengan magnitude besar (income: 1000-80000) akan **dominate** features dengan magnitude kecil (gender: 0-1)
- Without scaling: model basically ignores small-scale features
- Dengan scaling: semua features contribute equally

**fit() vs transform():**
- **scaler.fit(X_train)**: Learn μ dan σ dari training data only
- **scaler.transform(X_val)**: Apply same μ dan σ ke validation data
- **CRITICAL**: Never fit scaler pada validation/test data (causes data leakage)

---

## Model Training

### 7. Support Vector Machine (SVM)

```python
model = SVC(kernel='rbf')
model.fit(X, Y)

print("\n===== RESULTS =====")
print("Training ROC AUC:", roc_auc_score(Y, model.predict(X)))
print("Validation ROC AUC:", roc_auc_score(Y_val, model.predict(X_val)))
```

**Penjelasan SVM:**
- **Support Vector Machine**: Algoritma supervised learning untuk classification
- **Core Concept**: Mencari hyperplane optimal yang memisahkan dua kelas dengan **maximum margin**
- **Support Vectors**: Data points terdekat dengan decision boundary yang menentukan hyperplane
- **Margin**: Jarak dari hyperplane ke support vectors

**RBF Kernel:**
- **Radial Basis Function**: K(x, x') = exp(-γ ||x - x'||²)
- Maps data ke **higher dimensional space**
- Dapat handle **non-linear decision boundaries**
- Parameter γ (gamma) controls influence radius
  - High γ: complex boundary, risk of overfitting
  - Low γ: smooth boundary, might underfit

**Training Process:**
1. Solve quadratic programming optimization problem
2. Find optimal hyperplane coefficients (weights)
3. Identify support vectors
4. Model stores only support vectors (memory efficient)

**ROC AUC Score:**
- **ROC**: Receiver Operating Characteristic curve
- **AUC**: Area Under Curve
- **Interpretasi**:
  - 0.5 = Random classifier (coin flip)
  - 0.5-0.7 = Poor
  - 0.7-0.8 = Acceptable
  - 0.8-0.9 = Excellent
  - 0.9-1.0 = Outstanding
  - 1.0 = Perfect classifier

**Hasil:**
- **Training ROC AUC**: ~0.63 (menunjukkan performa pada training data)
- **Validation ROC AUC**: ~0.48 (performa pada unseen data)
- Gap antara training dan validation indicates **slight overfitting**
- Validation score lebih penting karena reflects real-world performance

---

## Model Evaluation

### 8. Confusion Matrix

```python
cm = confusion_matrix(Y_val, model.predict(X_val))
plt.figure(figsize=(6, 6))
sb.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
```

**Output**



<img width="936" height="448" alt="image" src="https://github.com/user-attachments/assets/a300b2fd-b1ce-45b8-b521-4ce771500229" />






**Penjelasan Confusion Matrix:**

|  | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | True Negative (TN) | False Positive (FP) |
| **Actual Positive** | False Negative (FN) | True Positive (TP) |

**Komponen:**
- **True Positive (TP)**: Prediksi approved, actual approved ✓
- **True Negative (TN)**: Prediksi rejected, actual rejected ✓
- **False Positive (FP)**: Prediksi approved, actual rejected ✗ (Type I Error)
- **False Negative (FN)**: Prediksi rejected, actual approved ✗ (Type II Error)

**Insight dari Gambar 6:**
- Confusion matrix memberikan **detailed breakdown** dari predictions
- Dari matrix bisa calculate: Accuracy, Precision, Recall, F1-Score
- **Business Impact**:
  - FP (approve bad loan) → financial loss untuk bank
  - FN (reject good applicant) → lost business opportunity

---

### 9. Classification Report

```python
print("\nClassification Report:")
print(classification_report(Y_val, model.predict(X_val)))
```

**Metrics Explained:**

**1. Precision = TP / (TP + FP)**
- "Of all predicted positives, berapa yang truly positive?"
- High precision = fewer false positives
- Penting untuk: fraud detection, spam filtering
- Loan context: Precision 0.67 = 67% approved loans are truly good

**2. Recall = TP / (TP + FN)**
- "Of all actual positives, berapa yang successfully detected?"
- High recall = fewer false negatives
- Penting untuk: disease screening, security threats
- Loan context: Recall varies per class

**3. F1-Score = 2 × (Precision × Recall) / (Precision + Recall)**
- Harmonic mean dari precision dan recall
- Balances trade-off antara precision dan recall
- Range: 0 to 1 (higher is better)
- Good for imbalanced datasets

**4. Support**
- Jumlah actual occurrences dari each class
- Menunjukkan distribusi data di validation set

**5. Accuracy = (TP + TN) / Total**
- Overall correctness rate
- Bisa misleading untuk imbalanced data!
- Example: 95% negative class → predict all negative = 95% accuracy tapi useless

**Hasil Model:**
- **Class 0 (Rejected)**: Precision 0.30, Recall 0.30 → Model struggles
- **Class 1 (Approved)**: Precision 0.67, Recall 0.67 → Better but not great
- **Overall Accuracy**: ~55% (barely better than random)
- **Conclusion**: Model needs improvement

---

## Hasil dan Analisis

### Performance Summary

| Metric | Score | Interpretation |
|--------|-------|----------------|
| Training ROC AUC | 0.63 | Acceptable |
| Validation ROC AUC | 0.48 | Poor (below random) |
| Overall Accuracy | 55% | Barely better than coin flip |
| Class 0 Precision | 0.30 | Poor |
| Class 1 Precision | 0.67 | Acceptable |
| Class 0 Recall | 0.30 | Poor |
| Class 1 Recall | 0.67 | Acceptable |

### Insights

**Yang Berhasil:**
1. Pipeline ML lengkap berhasil diimplementasikan
2. Handling imbalanced data dengan oversampling
3. Proper preprocessing (encoding, scaling, outlier removal)
4. Comprehensive evaluation dengan multiple metrics

**Challenges:**
1. Validation AUC (0.48) lebih rendah dari training (0.63) → overfitting
2. Overall accuracy 55% masih rendah
3. Class 0 (rejected loans) sulit diprediksi dengan baik
4. Gap besar antara training dan validation performance

**Possible Reasons:**
1. **Limited features** (hanya 5 features untuk prediction)
2. **Small dataset** (598 rows, after cleaning ~580 rows)
3. **High noise** dalam data
4. **Missing important features** (credit history, employment stability, existing debts)
5. **Wrong algorithm choice** (SVM mungkin bukan yang terbaik untuk data ini)

---

## Future Improvements

### 1. Feature Engineering
- **Create new features**:
  - Debt-to-Income Ratio = LoanAmount / ApplicantIncome
  - Loan-to-Income Percentage
  - Binary flag: High Income (> median)
- **Interaction features**: Gender × Married
- **Polynomial features**: Income², Income³

### 2. Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X, Y)
```

### 3. Try Different Algorithms
- **Random Forest**: Handles non-linear relationships, feature importance
- **XGBoost / LightGBM**: Excellent performance, handles missing values
- **Logistic Regression**: Simple baseline, interpretable
- **Neural Networks**: For complex patterns
- **Ensemble Methods**: Combine multiple models

### 4. Cross-Validation
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, Y, cv=5, scoring='roc_auc')
print(f"Cross-validation ROC AUC: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

### 5. Feature Selection
```python
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=3)
X_selected = selector.fit_transform(X, Y)
```

### 6. Handle Missing Values Explicitly
```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
df_imputed = imputer.fit_transform(df)
```

### 7. More Data
- Collect more samples (goal: >1000 rows)
- Add more features (credit score, employment history, etc.)
- Balance data collection (equal approved/rejected)

---

## Installation & Usage

### Prerequisites
```bash
Python 3.7+
pip
```

### Installation

1. **Clone repository**
```bash
git clone https://github.com/username/loan-prediction.git
cd loan-prediction
```

2. **Create virtual environment** (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### requirements.txt
```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
imbalanced-learn>=0.9.0
```

### Run the Script

```bash
python loan_prediction.py
```



---

*Last Updated: January 2026*

# 🧠 Obesity Level Prediction using Deep Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?style=for-the-badge&logo=keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellow?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Accuracy](https://img.shields.io/badge/Best%20Accuracy-94.09%25-brightgreen?style=for-the-badge)

<br/>

> **Predicting obesity levels in individuals from Mexico, Peru, and Colombia using hybrid deep learning architectures — achieving up to 94.09% accuracy across 7 obesity classes.**

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Model Architectures](#-model-architectures)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Discussion](#-discussion)
- [Author](#-author)

---

## 🔍 Overview

Obesity is a pressing global health concern linked to numerous physical and mental health complications. As its prevalence continues to rise, developing accurate prediction models becomes increasingly critical.

This project applies **three novel hybrid deep learning architectures** to predict an individual's obesity level based on their **eating habits and physical condition**. All models classify individuals into **7 distinct obesity categories** — from Insufficient Weight all the way to Obesity Type III.

### 🎯 Obesity Classes (Target Variable)
| Class | Label |
|-------|-------|
| 1 | Insufficient Weight |
| 2 | Normal Weight |
| 3 | Overweight Level I |
| 4 | Overweight Level II |
| 5 | Obesity Type I |
| 6 | Obesity Type II |
| 7 | Obesity Type III |

---

## 📊 Dataset

The dataset consists of **2,111 records** and **17 attributes** collected from individuals in **Mexico, Peru, and Colombia**.

> 📝 77% of data was synthetically generated using the **Weka tool** and **SMOTE filter** to balance the class distribution.

### 🔢 Features

| Feature | Type | Description |
|---------|------|-------------|
| `Gender` | Categorical | Male / Female |
| `Age` | Continuous | Age of individual |
| `Height` | Continuous | Height (meters) |
| `Weight` | Continuous | Weight (kg) |
| `family_history_with_overweight` | Binary | Family member with overweight history? |
| `FAVC` | Binary | Eats high-caloric food frequently? |
| `FCVC` | Integer | Frequency of vegetable consumption |
| `NCP` | Continuous | Number of main meals per day |
| `CAEC` | Categorical | Eating between meals |
| `SMOKE` | Binary | Smoker? |
| `CH2O` | Continuous | Daily water intake (liters) |
| `SCC` | Binary | Monitors calories? |
| `FAF` | Continuous | Physical activity frequency |
| `TUE` | Integer | Time using technological devices (hours) |
| `CALC` | Categorical | Alcohol consumption frequency |
| `MTRANS` | Categorical | Primary transportation mode |
| `NObeyesdad` | Categorical | **Target — Obesity Level** |

---

## 🏗️ Model Architectures

Three custom hybrid deep learning models were designed and evaluated:

### 1. 🔀 Hybrid Model — FNN + GRU

```
Categorical Features ──► GRU (32 units) ──┐
                                           ├──► Dense(64) ──► Softmax(7)
Numerical Features  ──► FNN (64→64)  ──────┘
```

- **GRU** processes one-hot encoded categorical features (reshaped to 3D for sequential compatibility)
- **FNN** processes standardized numerical features
- Outputs are concatenated and passed to a final classification layer

---

### 2. 🔁 Stacked Model — FNN + GRU

```
Categorical Features ──► GRU ──┐
                                ├──► Dense(64→32) ──► Stage 2 FNN ──► Softmax(7)
Numerical Features  ──► FNN ───┘
```

- Same as the hybrid model but uses a **two-stage approach**: the first stage extracts features, and the second stage re-classifies from those features

---

### 3. 🧱 Hybrid Model — ResNet + FNN

```
Categorical Features ──► Dense(64) ──► Dense(64) ──► Residual Add ──┐
                                                                      ├──► Dense(64→32) ──► Softmax(7)
Numerical Features  ──► Dense(64) ──► Dense(64) ────────────────────┘
```

- **ResNet-like architecture** with residual/skip connections for categorical features
- Helps prevent vanishing gradients and captures complex feature interactions

---

## 📈 Results

All three models were evaluated on a **20% test split** using the following metrics:

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| **Hybrid (FNN + GRU)** | **94.09%** | **93.98%** | **93.79%** | **93.87%** | **99.38%** |
| Stacked (FNN + GRU) | 94.02% | 93.87% | 93.93% | 93.84% | 99.33% |
| Hybrid (ResNet + FNN) | 94.02% | 93.87% | 93.93% | 93.84% | 99.33% |

### 🏆 Best Model — Hybrid FNN + GRU Confusion Matrix

```
               Predicted →
Actual ↓   [0]  [1]  [2]  [3]  [4]  [5]  [6]
  [0]       54    2    0    0    0    0    0
  [1]        2   55    0    0    0    4    1
  [2]        0    0   76    2    0    0    0
  [3]        0    0    1   57    0    0    0
  [4]        0    0    0    0   63    0    0
  [5]        0    7    0    0    0   48    1
  [6]        0    0    1    0    0    4   45
```

> ✅ The model performs exceptionally well across all 7 classes with minimal misclassification.

---

## 📁 Project Structure

```
obesity-level-prediction/
│
├── obesity_prediction.ipynb      # Main Jupyter notebook with all models
├── ObesityDataSet.csv            # Dataset (Mexico, Peru, Colombia)
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/obesity-level-prediction.git
cd obesity-level-prediction

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Requirements

```txt
tensorflow>=2.10
scikit-learn>=1.0
pandas>=1.4
numpy>=1.21
matplotlib>=3.5
seaborn>=0.11
```

---

## 🚀 Usage

```bash
# Open the Jupyter notebook
jupyter notebook obesity_prediction.ipynb
```

The notebook is organized into the following sections:

1. **Data Loading & Exploration** — Load the dataset and inspect its structure
2. **Exploratory Data Analysis** — Visualize feature distributions and relationships with obesity levels
3. **Data Preprocessing** — One-hot encoding, label encoding, train-test split, feature scaling
4. **Model 1: Hybrid FNN + GRU** — Build, train, and evaluate
5. **Model 2: Stacked FNN + GRU** — Build, train, and evaluate
6. **Model 3: Hybrid ResNet + FNN** — Build, train, and evaluate
7. **Comparison & Discussion** — Compare all models side-by-side

---

## 💬 Discussion

### ✅ Strengths
- All three architectures achieved **>94% accuracy** on a 7-class classification problem
- The **AUC-ROC score of 99.38%** demonstrates excellent class separability
- Hybrid approach effectively leverages both GRU's sequential learning and FNN's tabular strength

### ⚠️ Limitations
- Limited insight into **feature importance** (black-box nature of deep learning)
- Potential **sampling bias** — majority of data is synthetically generated
- Models may not generalize perfectly to other geographic regions

### 🔮 Future Improvements
- Incorporate **attention mechanisms** for better interpretability
- Experiment with **ensemble methods** (XGBoost + Neural Net)
- Apply **SHAP values** for feature importance analysis
- Extend dataset to cover more countries and demographics

---

## 👨‍💻 Author

**Usman Ghani**  
📚 CS251 — Introduction to Artificial Intelligence  
🏫 Semester Project | Instructor: Mam Nazia Shehzadi  
🎓 Registration No: 2022613

---

<div align="center">

⭐ **If you found this project helpful, please consider giving it a star!** ⭐

</div>

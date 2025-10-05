# 🧪 Offset Fault Detection — Air Quality Sensor Classification (LightGBM)

Binary classification project predicting **offset faults** in air-quality sensors using **LightGBM** and **time-aware validation**.  
The notebook covers complete **feature engineering**, **model training**, **evaluation**, and **submission generation**.

---

## 📘 Overview

This project was developed as part of the **KAUST vs KKU Machine Learning Tournament (Round 4)**.  
The goal is to detect whether an air-quality sensor reading is **faulty (1)** or **normal (0)** using data from multiple PM2.5 sensors and weather measurements.

---

## 📊 Dataset Description

Two datasets were provided:

- **Training Data (`train.csv`)**
  - Includes sensor readings, weather features, timestamp, and fault label (`Offset_fault`)
- **Testing Data (`test.csv`)**
  - Includes the same features **without labels**

| Column | Description |
|--------|--------------|
| `Datetime` | Timestamp of the recorded measurement |
| `Sensor1_PM2.5`, `Sensor2_PM2.5` | PM2.5 readings from two sensors |
| `Temperature` | Ambient temperature (°C) |
| `Relative_Humidity` | Air humidity (%) |
| `Offset_fault` | Target label (1 = Fault, 0 = Normal) *(train only)* |

---

## 🧠 Objective

Predict the **Offset_fault** (0 or 1) for each test record based on:
- Sensor discrepancies  
- Environmental factors  
- Temporal patterns (time of day, night vs. day)

---

## 🧰 Feature Engineering

- **Datetime Decomposition**
  - `Year`: extracted from timestamp  
  - `Hour`: hour of the day (0–23)  
  - `IsNight`: 1 if between 00:00–06:00  
- **Sensor Relationship**
  - `PM2.5_diff = abs(Sensor1_PM2.5 - Sensor2_PM2.5)`  
- **Preprocessing**
  - Dropped `Datetime` column after extraction  
  - Mean imputation for all missing values  
  - Removed `ID` column from test set before prediction  

Final feature set: [‘Sensor1_PM2.5’, ‘Sensor2_PM2.5’, ‘Temperature’, ‘Relative_Humidity’,
‘Year’, ‘Hour’, ‘IsNight’, ‘PM2.5_diff’]

---

## 📈 Correlation Heatmap

A heatmap was plotted to visualize relationships between variables and the target feature.

<img width="630" height="557" alt="image" src="https://github.com/user-attachments/assets/e9da9d37-4289-4a71-9f2b-5aa0168b402c" />


---

## ⚙️ Model and Training

### **Model Used:** LightGBM (LGBMClassifier)
```python
LGBMClassifier(
    objective='binary',
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    is_unbalance=True,
    random_state=42,
    n_jobs=-1
)
Validation Strategy
	•	TimeSeriesSplit (n_splits=5) used to prevent data leakage.
	•	Out-of-fold (OOF) predictions tracked for each split.
	•	Metric used: F1 Score (focuses on imbalanced binary classes).

📊 Performance

Metric
Score
OOF F1 Score
0.8523
✅ Strong generalization on unseen time-based validation sets.
🧩 Key Insights
	•	Temporal patterns (especially nighttime hours) correlate with higher fault probability.
	•	The difference between PM2.5 sensors (PM2.5_diff) is a strong indicator of faulty readings.
	•	Environmental conditions like humidity and temperature have moderate influence.
	•	Using time-aware cross-validation ensured the model didn’t overfit on chronological data.

⸻

✅ Summary

This project demonstrates a clean end-to-end machine learning pipeline:
	•	Comprehensive data preprocessing and feature extraction
	•	Robust time-aware validation (TimeSeriesSplit)
	•	Strong-performing LightGBM model achieving 0.85+ F1

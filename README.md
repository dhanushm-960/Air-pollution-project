# 🌍 Air Pollution Prediction and Safety Classification (Delhi)

## 📌 Overview

This project builds a machine learning system to:

* Predict the **Air Quality Index (AQI) for the next day**
* Classify whether the air quality will be **Safe or Dangerous**

The system uses **historical AQI trends and contextual features** instead of direct pollutant values, making it more realistic and applicable for decision-making.

---

## 🎯 Objectives

* Predict **future AQI values** using regression models
* Classify **air quality safety (Safe/Dangerous)**
* Use **feature engineering** instead of direct AQI formula recreation
* Build a **practical ML system for real-world use**

---

## 📊 Dataset

* File: `city_day.csv`
* Contains daily air quality data for Indian cities

### Data Used

* Only **Delhi** data is selected
* Each row represents **one day**

---

## 🧠 Feature Engineering

Instead of directly using pollutant values, the project uses derived features:

### ⏱ Time Features

* Month
* Day of week

### 🌦 Season Feature

* Winter, Summer, Monsoon, Post-monsoon

### 🚗 Traffic Feature

* Weekdays → High traffic
* Weekends → Low traffic

### 🏭 Industrial Activity

* Simulated using season (higher in winter)

### 🔁 Lag Features (Most Important)

* AQI_prev_1
* AQI_prev_2
* AQI_prev_3

---

## 🎯 Target Variables

### 🔵 Regression Target

* **AQI_next_day** → Predict next day AQI

---

### 🟢 Classification Target

* **Safety_Label**

  * 1 → Dangerous
  * 0 → Safe

👉 Important:
AQI is used **only to create labels**, not as input to the classification model.

---

## ⚙️ Models Used

### Regression

* Linear Regression
* KNN Regression

### Classification

* Logistic Regression
* KNN Classification

---

## 🔗 Project Pipeline

### 🔵 Regression Pipeline

```
Features → Regression Model → AQI_next_day
```

---

### 🟢 Classification Pipeline

```
Features → Classification Model → Safe / Dangerous
```

👉 Classification does **not use AQI directly**, it learns from patterns in features.

---

## 📈 Evaluation Metrics

### Regression

* RMSE
* MAE
* R² Score

---

### Classification

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

## 📊 Visualizations

### Regression

* Actual vs Predicted AQI (with reference line)
* Error distribution

### Classification

* Confusion matrix
* Accuracy comparison (Logistic vs KNN)

---

## 📊 Results Summary

### Regression

* Linear Regression performed best
* R² ≈ 0.89 (high accuracy)
* Average error ≈ 28 AQI units

---

### Classification

* KNN performed better than Logistic Regression
* Accuracy ≈ 65–68%
* Able to reasonably detect dangerous air conditions

---

## 🧠 Key Improvements

* Removed direct pollutant-based prediction
* Introduced feature engineering
* Shifted to **future AQI prediction**
* Redesigned classification to **Safe vs Dangerous**
* Built a **true ML model instead of rule-based thresholding**

---

## ⚠️ Important Concept

Although safety labels are created using an AQI threshold:

```
AQI > 200 → Dangerous
```

👉 The model **does NOT use this rule directly**.
Instead, it learns patterns from features such as past AQI, season, and traffic.

---

## 🌍 Applications

* Public health awareness (mask recommendations)
* Smart city monitoring
* Government decision systems
* Pollution alerts

---

## ⚠️ Limitations

* Daily data (not hourly)
* Some features are simulated
* Classification accuracy is moderate

---

## 🚀 Future Improvements

* Use real-time AQI data
* Add weather features (temperature, humidity, wind)
* Improve classification models
* Deploy as real-time system

---

## ▶️ How to Run

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Run the project:

```
python main.py
```

---

## 💡 Summary

This project predicts next-day AQI and classifies whether air quality will be safe or dangerous using machine learning and feature engineering.

---



# 🌍 Air Pollution Prediction and AQI Trend Analysis (Delhi)

## 📌 Overview

This project builds a machine learning system to:

* Predict the **Air Quality Index (AQI) for the next day**
* Determine whether air pollution will **increase or decrease**

The project focuses on creating a **realistic, data-driven pipeline** using historical AQI trends and contextual features.

---

## 🎯 Objectives

* Predict **future AQI values** using regression models
* Classify **pollution trends** (increase/decrease)
* Use **feature engineering** instead of direct pollutant data
* Build a **connected ML pipeline** (regression → classification)

---

## 📊 Dataset

* File: `city_day.csv`
* Source: Air quality data for Indian cities

### Data Used

* Only **Delhi** data is selected
* Each row represents **daily AQI information**

---

## 🧠 Feature Engineering

Instead of using pollutant values directly, the following features are created:

### ⏱ Time Features

* Month
* Day of week

### 🌦 Season Feature

* Winter, Summer, Monsoon, Post-monsoon

### 🚗 Traffic Feature

* Weekdays → High traffic
* Weekends → Low traffic

### 🏭 Industrial Activity

* Higher in winter

### 🔁 Lag Features

* AQI_prev_1
* AQI_prev_2
* AQI_prev_3

---

## 🎯 Target Variables

### 🔵 Regression

* **AQI_next_day** → Predict next day's AQI

### 🟢 Classification

* **Trend_Label**

  * 1 → AQI increases
  * 0 → AQI decreases

---

## ⚙️ Models Used

### Regression Models

* Linear Regression
* KNN Regression

### Classification Models

* Logistic Regression
* KNN Classification

---

## 🔗 Project Pipeline

```
Raw Features
   ↓
Feature Engineering
   ↓
Regression Model (Predict AQI_next_day)
   ↓
Classification Model (Predict Trend)
```

👉 Classification uses **predicted AQI**, not raw features.

---

## 📈 Evaluation Metrics

### Regression

* RMSE
* MAE
* R² Score

### Classification

* Accuracy
* Precision
* Recall
* F1 Score

---

## 📊 Results

### Regression

* Linear Regression performed best
* R² ≈ 0.89
* Average error ≈ 28 AQI units

### Classification

* KNN Classification performed best
* Accuracy ≈ 67%

---

## 🧠 Key Improvements

* Removed direct pollutant-based prediction
* Introduced feature engineering
* Shifted from current AQI to **future AQI prediction**
* Redesigned classification to predict **trend instead of category**
* Built a **connected pipeline** instead of independent models

---

## 🌍 Applications

* Pollution monitoring systems
* Government planning
* Smart city systems
* Public health alerts

---

## ⚠️ Limitations

* Uses daily data (not hourly)
* Some features are simulated (traffic, industry)
* Classification accuracy is moderate

---

## 🚀 Future Work

* Use real-time or hourly data
* Include weather data
* Improve classification accuracy
* Add real traffic and industrial datasets

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

This project predicts next-day AQI in Delhi and determines whether pollution will increase or decrease using historical trends and contextual features.

---

## 👨‍💻 Author

Dhanush M

---

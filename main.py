from src.preprocessing import load_data, filter_city, clean_data
from src.feature_engineering import *
from src.regression import run_regression
from src.classification import run_classification

from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

# 1️⃣ Load data
df = load_data("data/city_day.csv")

# 2️⃣ Preprocess
df = filter_city(df)
df = clean_data(df)

# 3️⃣ Feature Engineering
df = add_time_features(df)
df = add_season(df)
df = add_traffic(df)
df = add_industry(df)
df = add_lag_features(df)

# 4️⃣ Create targets
df['AQI_next_day'] = df['AQI'].shift(-1)

# 🔥 Classification target (Safe / Dangerous)
df['Safety_Label'] = df['AQI_next_day'].apply(
    lambda x: 1 if x > 200 else 0
)

# 5️⃣ Final processing
df = final_processing(df)

# 6️⃣ Prepare regression data
X = df.select_dtypes(include=['number'])
X = X.drop(['AQI','AQI_next_day','Safety_Label'], axis=1)

y_reg = df['AQI_next_day']

# 7️⃣ Train-test split (Regression)
X_train, X_test, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

# 🔵 Run Regression
run_regression(X_train, X_test, y_train_reg, y_test_reg)

# ==============================
# 🟢 CLASSIFICATION PART (FIXED)
# ==============================

# Use all features
X_class = X
y_class = df['Safety_Label']

# 🔥 Train-test split (IMPORTANT FIX)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42
)

# 🟢 Run Classification
run_classification(X_train_c, X_test_c, y_train_c, y_test_c)

# ==============================
# 📊 EXTRA GRAPH
# ==============================

plt.figure()
plt.plot(df['AQI'])
plt.title("AQI Trend Over Time")
plt.xlabel("Time")
plt.ylabel("AQI")
plt.show()
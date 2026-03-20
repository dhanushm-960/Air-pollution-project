from src.preprocessing import load_data, filter_city, clean_data
from src.feature_engineering import *
from src.regression import run_regression
from src.classification import run_classification

from sklearn.model_selection import train_test_split
import pandas as pd

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

# 4️⃣ Create Targets
df['AQI_next_day'] = df['AQI'].shift(-1)

df['AQI_trend'] = df['AQI_next_day'] - df['AQI']
df['Trend_Label'] = df['AQI_trend'].apply(lambda x: 1 if x > 0 else 0)

# 5️⃣ Final Processing
df = final_processing(df)

# 🔥 6️⃣ KEEP ONLY NUMERIC FEATURES (IMPORTANT FIX)
X = df.select_dtypes(include=['number'])

# Remove target columns
X = X.drop(['AQI','AQI_next_day','AQI_trend','Trend_Label'], axis=1)

# Regression target
y_reg = df['AQI_next_day']

# 7️⃣ Train-Test Split
X_train, X_test, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

# 🔵 REGRESSION
y_pred_reg = run_regression(X_train, X_test, y_train_reg, y_test_reg)

# 🟢 CLASSIFICATION INPUT (NEW PIPELINE)

# Previous AQI values
AQI_prev_test = df.loc[X_test.index, 'AQI_prev_1']

# Build classification dataset
X_class = pd.DataFrame({
    'Predicted_AQI': y_pred_reg,
    'AQI_prev': AQI_prev_test
})

# Classification target
y_class = df.loc[X_test.index, 'Trend_Label']

# 🟢 CLASSIFICATION
run_classification(X_class, y_class)
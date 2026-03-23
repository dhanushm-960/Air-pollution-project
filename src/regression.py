from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt


def run_regression(X_train, X_test, y_train, y_test):

    print("\n--- Linear Regression ---")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
    print("MAE:", mean_absolute_error(y_test, y_pred_lr))
    print("R2:", r2_score(y_test, y_pred_lr))

    print("\n--- KNN Regression ---")

    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)

    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_knn)))
    print("MAE:", mean_absolute_error(y_test, y_pred_knn))
    print("R2:", r2_score(y_test, y_pred_knn))

    # ============================
    # 📊 PLOT 1: Actual vs Predicted
    # ============================

    plt.figure()
    plt.scatter(y_test, y_pred_lr)

    # 🔥 Best-fit reference line (y = x)
    min_val = min(y_test.min(), y_pred_lr.min())
    max_val = max(y_test.max(), y_pred_lr.max())

    plt.plot([min_val, max_val],
             [min_val, max_val],
             linestyle='--')

    plt.xlabel("Actual AQI")
    plt.ylabel("Predicted AQI")
    plt.title("Actual vs Predicted AQI (Linear Regression)")
    plt.show()

    # ============================
    # 📊 PLOT 2: Error Distribution
    # ============================

    errors = y_test - y_pred_lr

    plt.figure()
    plt.hist(errors)
    plt.title("Error Distribution (Linear Regression)")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.show()

    return y_pred_lr
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

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

    return y_pred_lr
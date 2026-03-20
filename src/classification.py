from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

def run_classification(X_class, y_class):

    print("\n--- Logistic Regression ---")

    log = LogisticRegression(max_iter=1000)
    log.fit(X_class, y_class)
    y_pred = log.predict(X_class)

    print("Accuracy:", accuracy_score(y_class, y_pred))
    print(classification_report(y_class, y_pred))

    print("\n--- KNN Classification ---")

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_class, y_class)
    y_pred_knn = knn.predict(X_class)

    print("Accuracy:", accuracy_score(y_class, y_pred_knn))
    print(classification_report(y_class, y_pred_knn))
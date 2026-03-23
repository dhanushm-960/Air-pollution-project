from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

def run_classification(X_train, X_test, y_train, y_test):

    print("\n--- Logistic Regression ---")

    log = LogisticRegression(max_iter=1000)
    log.fit(X_train, y_train)
    y_pred = log.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    print("\n--- KNN Classification ---")

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred_knn))
    print(classification_report(y_test, y_pred_knn))

    # 📊 Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.text(j, i, cm[i][j])

    plt.show()

    # 📊 Accuracy Comparison
    log_acc = accuracy_score(y_test, y_pred)
    knn_acc = accuracy_score(y_test, y_pred_knn)

    plt.figure()
    models = ['Logistic Regression', 'KNN']
    accuracy = [log_acc, knn_acc]

    plt.bar(models, accuracy)
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.show()
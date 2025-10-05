# src/metrics.py
from sklearn.metrics import classification_report, confusion_matrix

def summarize(y_true, y_pred):
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nReport:")
    print(classification_report(y_true, y_pred, digits=4))

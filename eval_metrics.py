import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn import metrics


def auc_score(y_true, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_true=y_true, y_score=y_pred, pos_label=1)
    return metrics.auc(fpr, tpr)


if __name__ == "__main__":
    # a = [0, 0, 0, 0]
    a = [1, 1, 1, 1]
    b = [0, 1, 0, 1]
    print accuracy_score(y_true=a, y_pred=b)
    print precision_score(y_true=a, y_pred=b)
    print recall_score(y_true=a, y_pred=b)
    print f1_score(y_true=a, y_pred=b)
    print auc_score(y_true=a, y_pred=b)
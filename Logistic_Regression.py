from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_logistic_regression(X_train_lr, X_test_lr, y_train_lr, y_test_lr):

    penalties = ['l2', None]
    results = []
    

    for penalty in penalties:

        lr = LogisticRegression(penalty=penalty)
        lr.fit(X_train_lr, y_train_lr)
        y_pred_train_lr = lr.predict(X_train_lr)
        y_pred_test_lr = lr.predict(X_test_lr)

        results.append({
            'model': penalty,
            'train_accuracy': accuracy_score(y_train_lr, y_pred_train_lr),
            'test_accuracy': accuracy_score(y_test_lr, y_pred_test_lr),
            'confusion_matrix': confusion_matrix(y_test_lr, y_pred_test_lr),
            'classification_report': classification_report(y_test_lr,y_pred_test_lr)
        })
    return results
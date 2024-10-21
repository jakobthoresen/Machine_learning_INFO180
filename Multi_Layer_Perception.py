from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def train_MLP(X_train_lr, X_test_lr, y_train_lr, y_test_lr):
    
    Architectures = [
        [15, 10, 6], #3 hidden layers
        [10, 4],      #2 hidden layers
        [10, 5, 2],   #3 hidden layers
        [20, 10, 5],  #3 hidden layers
    ]

    results = []

    for architecture in Architectures:

        MLP = MLPClassifier(hidden_layer_sizes=architecture, max_iter=1000, learning_rate='adaptive', random_state=42)
        
        MLP.fit(X_train_lr, y_train_lr)
        y_pred_train_lr = MLP.predict(X_train_lr)
        y_pred_test_lr = MLP.predict(X_test_lr)


        results.append({
            'model': architecture,
            'train_accuracy': accuracy_score(y_train_lr, y_pred_train_lr),
            'test_accuracy': accuracy_score(y_test_lr, y_pred_test_lr),
            'confusion_matrix': confusion_matrix(y_test_lr, y_pred_test_lr),
            'classification_report': classification_report(y_test_lr,y_pred_test_lr)
        })
    return results

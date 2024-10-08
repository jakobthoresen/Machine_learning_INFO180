from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_decision_tree(X_train, X_test, y_train, y_test):
    results = []

    dt_gini = DecisionTreeClassifier(criterion='gini', random_state=42)
    dt_gini.fit(X_train, y_train)
    y_pred_train = dt_gini.predict(X_train)
    y_pred_test = dt_gini.predict(X_test)

    #Decision Tree with Gini index
    results.append({
        'model': 'gini',
        'train_accuracy': accuracy_score(y_train, y_pred_train),
        'test_accuracy': accuracy_score(y_test, y_pred_test),
        'confusion_matrix': confusion_matrix(y_test, y_pred_test),
        'classification_report': classification_report(y_test,y_pred_test)
    })

    # Decision Tree with Entropy
    dt_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
    dt_entropy.fit(X_train, y_train)
    y_pred_train_entropy = dt_entropy.predict(X_train)
    y_pred_test_entropy = dt_entropy.predict(X_test)
    
    results.append({
        'model': 'entropy',
        'train_accuracy': accuracy_score(y_train, y_pred_train_entropy),
        'test_accuracy': accuracy_score(y_test, y_pred_test_entropy),
        'confusion_matrix': confusion_matrix(y_test, y_pred_test_entropy),
        'classification_report': classification_report(y_test, y_pred_test_entropy)
    })

    return results
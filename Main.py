import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from KNeighbors import train_knn
from Logistic_Regression import train_logistic_regression
from Decision_Tree import train_decision_tree



#Opening the data in pandas
def load_and_process_data(filename):
    data = pd.read_csv(filename)
    #data['ok guest'] = data['ok guest'].map({'ok':1, 'not ok':0})

    #One-hot encode for k-NN and Decision Tree
    data_encoded = pd.get_dummies(data, drop_first=False, columns=['gender','age', 'study', 'activity', 'music', 'is dancer'])

    #One-hot encode for Logistic regression
    data_encoded_lr = pd.get_dummies(data, drop_first=True, columns=['gender','age', 'study', 'activity', 'music', 'is dancer'])

    return data_encoded, data_encoded_lr


def split_data(data_encoded):

    X = data_encoded.drop('ok guest', axis=1)

    y = data_encoded['ok guest']
    return train_test_split(X, y, test_size=0.2, random_state=42)


def display_results(results, model_name):
    for result in results:
        print(f"{model_name} with parameters: {result['model']}")
        print(f'Training accuracy:', result['train_accuracy'])
        print(f'Testing accuracy:', result['test_accuracy'])
        print(f'Confusion Matrix:\n', result['confusion_matrix'])
        print()

def main():
    #Filepath to the dataset
    filename = 'party_data.csv'

    #Load and process the data
    data_encoded, data_encoded_lr = load_and_process_data(filename)

    #Split the data
    X_train, X_test, y_train, y_test = split_data(data_encoded)
    X_train_lr, X_test_lr, y_train_lr, y_test_lr = split_data(data_encoded_lr)

    # Train and evaluate k-NN
    knn_results = train_knn(X_train,X_test,y_train,y_test)
    display_results(knn_results, 'k-NN')

    #Train and evaluate Logistic Regression
    lr_results = train_logistic_regression(X_train_lr, X_test_lr, y_train_lr, y_test_lr)
    display_results(lr_results, 'Logistic Regression')

    #Train and evaluate Decision Tree
    dt_results = train_decision_tree(X_train, X_test, y_train, y_test)
    display_results(dt_results, 'Decision Tree')


if __name__ == '__main__':
    main()
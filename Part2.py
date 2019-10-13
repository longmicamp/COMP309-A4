# -*- coding: utf-8 -*-

"""
This is an example to perform simple linear regression algorithm on the dataset (weight and height),
where x = weight and y = height.
"""
import pandas as pd
import numpy as np

import random

from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score


from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# General settings


from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score

seed = 0
# Freeze the random seed
random.seed(seed)
np.random.seed(seed)
train_test_split_test_size = 0.1

# Training settings
alpha = 0.1  # step size
max_iters = 50  # max iterations


def load_data():
    """
    Load Data from CSV
    :return: df    a panda data frame
    """

    columnsName = ['age', "workClass", 'num', 'Education', 'num2', 'maritalStatus', 'occupation', 'relationship',
                   'race', 'sex', 'num3', 'num4', 'num5', 'country', 'salary']

    trainData = pd.read_csv('adult.data', header=None, names=columnsName, delimiter=' *, *')

    testData = pd.read_csv('adult.test', header=None, names=columnsName, delimiter=' *, *')

    return trainData, testData


def data_preprocess(testData, trainData):
    testData = testData.drop(["Education"], axis=1)
    trainData = trainData.drop(["Education"], axis=1)

    replace_categorical = {"workClass": {'Private': 1, 'Self-emp-not-inc': 2, 'Self-emp-inc': 3, 'Federal-gov': 4,
                                         'Local-gov': 5, 'State-gov': 6, 'Without-pay': 7, 'Never-worked': 8, '?': 1},
                           "maritalStatus": {'Married-civ-spouse': 1, 'Divorced': 2, 'Never-married': 3, 'Separated': 4,
                                             'Widowed': 5, 'Married-spouse-absent': 6, 'Married-AF-spouse': 7,
                                             'Never-worked': 8},
                           "occupation": {'Tech-support': 1, 'Craft-repair': 2, 'Other-service': 3, 'Sales': 4,
                                          'Exec-managerial': 5, 'Prof-specialty': 6, 'Handlers-cleaners': 7,
                                          'Machine-op-inspct': 8, 'Adm-clerical': 9, 'Farming-fishing': 10,
                                          'Transport-moving': 11, 'Priv-house-serv': 12, 'Protective-serv': 13,
                                          'Armed-Forces': 14, '?': 4},
                           "relationship": {'Wife': 1, 'Own-child': 2, 'Husband': 3, 'Not-in-family': 4,
                                            'Other-relative': 5, 'Unmarried': 6},
                           "race": {'White': 1, 'Asian-Pac-Islander': 2, 'Amer-Indian-Eskimo': 3, 'Other': 4,
                                    'Black': 5},
                           "sex": {'Female': 0, 'Male': 1},
                           "country": {'United-States': 1, 'Cambodia': 2, 'England': 3, 'Puerto-Rico': 4,
                                       'Canada': 5, 'Germany': 6, 'Outlying-US(Guam-USVI-etc)': 7, 'India': 8,
                                       'Japan': 9, 'Greece': 10, 'South': 11, 'China': 12, 'Cuba': 13, 'Iran': 14,
                                       'Honduras': 15, 'Philippines': 16, 'Italy': 17, 'Poland': 18,
                                       'Jamaica': 19, 'Vietnam': 20, 'Mexico': 21, 'Portugal': 22, 'Ireland': 23,
                                       'France': 24, 'Dominican-Republic': 25, 'Laos': 26, 'Ecuador': 27,
                                       'Taiwan': 28, 'Haiti': 29, 'Columbia': 30, 'Hungary': 31, 'Guatemala': 32,
                                       'Nicaragua': 33, 'Scotland': 34, 'Thailand': 35, 'Yugoslavia': 36,
                                       'El-Salvador': 37, 'Trinadad&Tobago': 38, 'Peru': 39, 'Hong': 40,
                                       'Holand-Netherlands': 41, '?': 1},
                           "salary": {'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1}}
    testData.replace(replace_categorical, inplace=True)
    trainData.replace(replace_categorical, inplace=True)

    train_data_full = trainData.copy()
    train_labels = train_data_full["salary"]
    train_data = trainData.drop(["salary"], axis=1)
    train_labels = train_labels.astype('int')

    test_data_full = testData.copy()
    test_labels = test_data_full["salary"]
    test_data = testData.drop(["salary"], axis=1)
    test_labels = test_labels.astype('int')
    pd.options.mode.chained_assignment = None


    train_data['intercept_dummy'] = pd.Series(1.0, index=train_data.index)
    test_data['intercept_dummy'] = pd.Series(1.0, index=test_data.index)

    return train_data, train_labels, test_data, test_labels, train_data_full, test_data_full


def learn():
    train_data1, test_data1 = load_data()
    train_data, train_labels, test_data, test_labels, train_data_full, test_data_full = data_preprocess(train_data1,
                                                                                                        test_data1)

    classify(test_data, test_labels, train_data, train_labels, GaussianNB)
    classify(test_data, test_labels, train_data, train_labels, KNeighborsClassifier)
    classify(test_data, test_labels, train_data, train_labels, svm.SVC)
    classify(test_data, test_labels, train_data, train_labels, tree.DecisionTreeClassifier)
    classify(test_data, test_labels, train_data, train_labels, RandomForestClassifier)
    classify(test_data, test_labels, train_data, train_labels, AdaBoostClassifier)
    classify(test_data, test_labels, train_data, train_labels, GradientBoostingClassifier)
    classify(test_data, test_labels, train_data, train_labels, LinearDiscriminantAnalysis)
    classify(test_data, test_labels, train_data, train_labels, MLPClassifier)
    classify(test_data, test_labels, train_data, train_labels, LogisticRegression)


def classify(testData, testLabels, trainData, trainLabels, function):
    reg = function().fit(trainData, trainLabels)
    predictedY = reg.predict(testData)

    score = r2_score(testLabels, predictedY)
    accuracy = accuracy_score(testLabels, predictedY)
    precision = precision_score(testLabels, predictedY, average='weighted')
    recall = recall_score(testLabels, predictedY, average='macro')
    f1score = f1_score(testLabels, predictedY, average='macro')
    auc = roc_auc_score(testLabels, predictedY)
    # plt.bar(y_pos, predictedY, color='r')
    # plt.ylabel("Predicted Price")
    # plt.xlabel("Actual Price value")
    # plt.title(function)
    # plt.show()

    print('------------------------')
    print(function)

    print('Accuracy: ', accuracy)
    print('precision: ', precision)
    print('recall: ', recall)
    print('AUC: ', auc)
    print('F1 Score: ', f1score)
    print('-------------------------')


learn()

# -*- coding: utf-8 -*-

"""
This is an example to perform simple linear regression algorithm on the dataset (weight and height),
where x = weight and y = height.
"""
import pandas as pd
import numpy as np
import datetime
import random
import sys
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPRegressor
from scipy import stats
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error



# General settings

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
from math import sqrt
from sklearn.model_selection import cross_val_score
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
    df = pd.read_csv("diamonds3.csv")

    return df


def data_preprocess(data):
    """
    Data preprocess:
        1. Split the entire dataset into train and test
        2. Split outputs and inputs
        3. Standardize train and test
        4. Add intercept dummy for computation convenience
    :param data: the given dataset (format: panda DataFrame)
    :return: train_data       train data contains only inputs
             train_labels     train data contains only labels
             test_data        test data contains only inputs
             test_labels      test data contains only labels
             train_data_full       train data (full) contains both inputs and labels
             test_data_full       test data (full) contains both inputs and labels
    """
    # Split the data into train and test
    train_data, test_data = train_test_split(data, test_size = 0.3, random_state=309)



    # Pre-process data (both train and test)
    train_data_full = train_data.copy()
    train_labels = train_data_full["price"]
    train_data = train_data.drop(["price"], axis =1)
    train_data = train_data.drop(["Unnamed: 0"], axis=1)
    test_data_full = test_data.copy()
    test_labels = test_data_full["price"]
    test_data = test_data.drop(["price"], axis=1)
    test_data = test_data.drop(["Unnamed: 0"], axis=1)

    # Standardize the inputs
    sns.set(color_codes=True)

    sns.distplot(test_data.x)
    plt.show()


    train_mean = train_data.mean()
    train_std = train_data.std()
    train_data = (train_data - train_mean) / train_std

    test_mean = test_data.mean()
    test_std = test_data.std()
    test_data = (test_data - test_mean) / test_std


    sns.distplot(test_data.x)
    plt.show()



    # Tricks: add dummy intercept to both train and test
    pd.options.mode.chained_assignment = None
    train_data['intercept_dummy'] = pd.Series(1.0, index = train_data.index)
    test_data['intercept_dummy'] = pd.Series(1.0, index = test_data.index)



    return train_data, train_labels, test_data, test_labels, train_data_full, test_data_full


def dataVisual(traindata):

    plt.scatter(traindata.x, traindata.y, s=np.pi * 3, c="red", alpha=0.5)
    plt.title("X vs Y")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()



def learn():


    # Step 1: Load Data
    data = load_data()

    # Step 2: Preprocess the data
    train_data, train_labels, test_data, test_labels, train_data_full, test_data_full = data_preprocess(data)
    classify(test_data,test_labels,train_data,train_labels,LinearRegression)
    classify(test_data, test_labels, train_data, train_labels,KNeighborsRegressor)
    classify(test_data, test_labels, train_data, train_labels, Ridge)
    classify(test_data, test_labels, train_data, train_labels, DecisionTreeRegressor)
    classify(test_data, test_labels, train_data, train_labels, RandomForestRegressor)
    classify(test_data, test_labels, train_data, train_labels, GradientBoostingRegressor)
    classify(test_data, test_labels, train_data, train_labels, linear_model.SGDRegressor)
    classify(test_data, test_labels, train_data, train_labels, SVR)
    classify(test_data, test_labels, train_data, train_labels, MLPRegressor)

def classify(testData,testLabels,trainData,trainLabels, function):
    start_time = datetime.datetime.now()
    reg = function().fit(trainData,trainLabels)
    predictedY = reg.predict(testData)
    end_time = datetime.datetime.now()
    mse = mean_squared_error(predictedY, testLabels)
    rmse = sqrt(mean_squared_error(predictedY, testLabels))
    score = r2_score(testLabels, predictedY)
    MAE = mean_absolute_error(testLabels, predictedY)
    exection_time = (end_time - start_time).total_seconds()



    plt.scatter(testLabels, predictedY, color='r')
    plt.ylabel("Predicted Price")
    plt.xlabel("Actual Price value")
    plt.title(function)
    plt.show()



    print('------------------------')
    print(function)
    print("Learn: execution time={t:.3f} seconds".format(t=exection_time))
    print('MAE: ', MAE)
    print('Mean squared error:', mse)
    print('Root Mean squared error:',rmse)
    print('R2 score: ', score)
    print('-------------------------')

learn()
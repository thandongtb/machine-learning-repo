import os
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

def getData():
    # Get home data from CSV file
    dataFile = None
    if os.path.exists('home_data.csv'):
        print("-- home_data.csv found locally")
        dataFile = pd.read_csv('home_data.csv', skipfooter=1)

    return dataFile

def linearRegressionModel(X_train, Y_train, X_test, Y_test):
    linear = linear_model.LinearRegression()

    # Training process
    linear.fit(X_train, Y_train)
    # Evaluating the model
    score_trained = linear.score(X_test, Y_test)

    return score_trained

def lassoRegressionModel(X_train, Y_train, X_test, Y_test):
    lasso_linear = linear_model.Lasso(alpha=1.0)
    # Training process
    lasso_linear.fit(X_train, Y_train)
    # Evaluating the model
    score_trained = lasso_linear.score(X_test, Y_test)

    return score_trained

def polynomialRegression(X_train, Y_train, X_test, Y_test, degree):
    poly_model = Pipeline([('poly', PolynomialFeatures(degree)),
                           ('linear', linear_model.LinearRegression(fit_intercept=False))])
    poly_model = poly_model.fit(X_train, Y_train)
    score_poly_trained = poly_model.score(X_test, Y_test)

    return score_poly_trained

if __name__ == "__main__":
    data = getData()
    if data is not None:
        # Selection few attributes
        attributes = list(
            [
                'num_bed',
                'year_built',
                'num_room',
                'num_bath',
                'living_area',
                'accessible_buildings',
                'family_quality',
                'art_expos'
            ]
        )
        # Vector price of house
        Y = data['askprice']
        # print np.array(Y)
        # Vector attributes of house
        X = data[attributes]
        # Split data to training test and testing test
        X_train, X_test, Y_train, Y_test = train_test_split(np.array(X), np.array(Y), test_size=0.2)
        # Linear Regression Model
        linearScore = linearRegressionModel(X_train, Y_train, X_test, Y_test)
        print 'Linear Score = ' , linearScore

        # LASSO Regression Model
        lassoScore = lassoRegressionModel(X_train, Y_train, X_test, Y_test)
        print 'Lasso Score = ', lassoScore

        # Poly Regression Model
        polyScore = polynomialRegression(X_train, Y_train, X_test, Y_test, 3)
        print 'Poly Score = ', polyScore

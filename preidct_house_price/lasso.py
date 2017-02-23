from sqlalchemy import *
from sklearn import cross_validation
import numpy as np
from sklearn.linear_model import Lasso
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn import datasets


def load_boston_data():
    """Load the Boston dataset."""

    boston = datasets.load_boston()
    return boston

def loadAbsArray(data):
    temp_data = []

    for raw in data:
        temp_raw = []

        for i in range(len(raw)):
            temp_raw.append(abs(float(raw[i])))

        temp_data.append(temp_raw)

    return temp_data

def loadAbsPrice(price):
    lenPrice = len(price)
    tempData = []

    for i in range(lenPrice):
        tempData.append(abs(float(price[i])))

    return tempData

def loadSqrtArray(data):
    lenX = len(data[0])
    print 'len X = ',lenX
    lenY = len(data)

    XX = []
    data = np.sqrt(data[:,:])

    for i in range(0,lenY):
        temp = []

        for k in range(0, lenX):
            for l in range(k, lenX):
                temp.append(data[i][k]*data[i][l])
        XX.append(temp)

    return np.array(XX)

def initIdtrIndex(k,mte,n):
    result = []
    k = int(k)
    mte = int(mte)
    n = int(n)

    for i in range(0,(k-1)*mte):
        result.append(i)

    for i in range(k*mte + 1, n):
        result.append(i)

    return np.array(result)

def sumTwoMatrix(a,b):
    return np.array(np.add(a,b))

def get_lasso_plot(lamda,mean_error,dld,maxder):
    xs = lamda
    ys_mean = mean_error
    plt.plot(xs, ys_mean, 'g-')
    plt.xlabel('lambda')
    plt.ylabel('Mean Absolute Error (Trieu VND)')
    title = 'Agregation LASSO Mean Error With dld = ',dld,' maxder = ',maxder
    plt.title(title)
    plt.legend(loc=1)
    plt.show()

def get_predictor_plot(lamda,heso,dld,maxder):
    x = lamda
    y = heso
    plt.xlabel('Lambda')
    plt.ylabel('Estimate Index')
    title = 'Agregation LASSO Estimate Index With dld = ',dld,' maxder = ',maxder
    plt.title(title)
    plt.plot(x, y, 'b-', x, y, 'ro')
    plt.grid(True)
    plt.show()

def get_result_plot(stt,root,result,dld,maxder):
    x = stt
    y1 = root
    y2 = result
    plt.xlabel('Order of Real Estate')
    plt.ylabel('Value')
    title = 'Agregation LASSO Result With dld = ',dld,' maxder = ',maxder
    plt.title(title)
    plt.plot(x, y1, 'c-', x, y1, 'red')
    plt.plot(x, y2, 'c-', x, y2, 'green')
    plt.grid(True)
    plt.show()

def main():
    boston = load_boston_data()
    price = boston.target
    data = boston.data
    d = len(data[0])
    n = len(data)
    p = d + 1
    data_abs = loadAbsArray(data)
    X = np.array(data)
    Y = np.array(price)

    # Absolute verto of X
    absX = np.array(loadAbsArray(X))
    absY = np.array(loadAbsPrice(Y))

    # Initial sqrt vector
    XX = loadSqrtArray(absX)
    print(X.shape[1])
    YY = np.sqrt(absY)

    # Initial Lasso Algorithms
    ld0 = 0
    dld = 0.0001
    maxder = 0.05

    kl = 1
    kF = 5

    # Tinh toan error0
    error0 = []
    accs = []
    kf_removed = cross_validation.KFold(YY.shape[0], n_folds=5)

    for train_index, test_index in kf_removed:
        X_train, X_test = XX[train_index], XX[test_index]
        Y_train, Y_test = YY[train_index], YY[test_index]
        clf = Lasso(ld0)
        clf.fit(X_train, Y_train)
        acc = clf.predict(X_train)
        error = mean_absolute_error(acc, Y_train)
        accs.append(error)

    error0 = np.mean(accs)
    print "Initial Lasso (lamda = 0):", ld0, error0

    # Aggregation Lasso Regression using the data set with missing data removed.
    er = error0
    lamda = []
    mean_er = []
    min_er = error0
    temp_lda = ld0

    while (er < error0 + maxder):
        print('The loop ', kl)

        if (np.floor(kl / 2) != kl / 2) and ((ld0 - (np.floor(kl / 2) + 1) * dld) >= 0):
            lda = ld0 - (np.floor(kl / 2) + 1) * dld
            lamda.append(lda)
        else:
            lda = ld0 + np.floor(kl / 2) * dld
            lamda.append(lda)
        kf_removed = cross_validation.KFold(YY.shape[0], n_folds=5)

        print "Lambda  = ", lda
        trer = []

        for train_index, test_index in kf_removed:
            X_train, X_test = XX[train_index], XX[test_index]
            Y_train, Y_test = YY[train_index], YY[test_index]
            lasso = Lasso(lda)
            lasso.fit(X_test, Y_test)
            acc = lasso.predict(X_test)
            error = mean_absolute_error(acc, Y_test)
            trer.append(error)
        er = np.mean(trer)

        if (er < min_er):
            min_er = er
            temp_lda = lda
        mean_er.append(er)

        print 'Error = ', er
        kl = kl + 1

    # get_lasso_plot(lamda,mean_er,dld,maxder)
    heso = []

    for l in lamda:
        lasso = Lasso(l)
        lasso.fit(XX, YY)
        heso.append(lasso.coef_)

    heso = np.array(heso)
    res_heso = []

    for key in range(0, heso.shape[1]):
        res_heso.append(np.mean(heso[:, key]))

    print(res_heso)
    res_index = []

    for i in range(0, len(XX[0])):
        res_index.append(i)
    # get_predictor_plot(np.array(res_index),res_heso,dld,maxder)

    result = np.dot(XX, np.array(res_heso))
    res_stt = []

    for i in range(0, np.array(np.square(YY)).shape[0]):
        res_stt.append(i)

    get_result_plot(np.array(res_stt), np.array(np.square(YY)), np.array(np.square(result)), dld, maxder)

    print 'result = ', np.square(result)
    print 'y = ', np.square(YY)
    res_percentage = []
    temp1 = np.array(np.square(YY))
    temp2 = np.array(np.square(result))

    for i in range(0, np.array(np.square(YY)).shape[0]):
        res_percentage.append(np.abs(temp1[i] - temp2[i]) / temp1[i])

    print(np.mean(res_percentage))

if __name__ == "__main__":
    main()

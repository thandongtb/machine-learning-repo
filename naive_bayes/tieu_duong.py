import csv
import random
import math
import sys

# Load data tu CSV file

def loadDataFromCsv(filename):
    lines = csv.reader(open(filename, "rb"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]

    return dataset

# Phan chia tap du lieu theo class

def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)

    return separated

# Phan chia tap du lieu thanh training va testing. Co the dung train_test_split

def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))

    return [trainSet, copy]

# tinh toan gia tri trung binh cua moi thuoc tinh

def mean(numbers):
    return sum(numbers) / float(len(numbers))

# Tinh toan do lech chuan cho tung thuoc tinh
# https://i.ytimg.com/vi/zaSt2WkP8eE/maxresdefault.jpg

def standardDeviation(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)

    return math.sqrt(variance)

# Gia tri trung binh , do lech chuan

def summarize(dataset):
    summaries = [(mean(attribute), standardDeviation(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]

    return summaries


def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.iteritems():
        summaries[classValue] = summarize(instances)

    return summaries

# Tinh toan xac suat theo phan phoi Gause cua bien lien tuc
# http://sites.nicholas.duke.edu/statsreview/files/2013/06/normpdf1.jpg

def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))

    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

# Tinh xac suat cho moi thuoc tinh phan chia theo class
def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.iteritems():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)

    return probabilities

# Du doan vector thuoc phan lop nao

def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.iteritems():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue

    return bestLabel

# Du doan tap du lieu training thuoc vao phan lop nao

def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)

    return predictions

# In ket qua phan lop

def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1

    return (correct / float(len(testSet))) * 100.0

def predictPreview(summaries, testSet, size):
    if (size <= len(testSet)):
        for i in range(size):
            print('\nTest[{0}] = {1}\nCorrect class = {2} Predict class = {3}\n').format(i, testSet[i], testSet[i][-1], predict(summaries, testSet[i]))

    return 0
def main():
    filename = 'tieu_duong.csv'
    splitRatio = 0.8
    dataset = loadDataFromCsv(filename)
    trainingSet, testSet = splitDataset(dataset, splitRatio)
    print('Data size {0} \nTraining Size={1} \nTest Size={2}').format(len(dataset), len(trainingSet), len(testSet))
    # prepare model
    summaries = summarizeByClass(trainingSet)
    predictPreview(summaries, testSet, 10)
    # test model
    predictions = getPredictions(summaries, testSet)
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: {0}%').format(accuracy)


if __name__ == "__main__":
    main()
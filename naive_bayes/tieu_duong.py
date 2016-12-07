import csv
import random
import math
import sys

# Load data tu CSV file

def load_data(filename):
    lines = csv.reader(open(filename, "rb"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]

    return dataset

# Phan chia tap du lieu theo class

def separate_data(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)

    return separated

# Phan chia tap du lieu thanh training va testing. Co the dung train_test_split

def split_data(dataset, splitRatio):
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

def standard_deviation(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)

    return math.sqrt(variance)

# Gia tri trung binh , do lech chuan

def summarize(dataset):
    summaries = [(mean(attribute), standard_deviation(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]

    return summaries


def summarize_by_class(dataset):
    separated = separate_data(dataset)
    summaries = {}
    for classValue, instances in separated.iteritems():
        summaries[classValue] = summarize(instances)

    return summaries

# Tinh toan xac suat theo phan phoi Gause cua bien lien tuc
# http://sites.nicholas.duke.edu/statsreview/files/2013/06/normpdf1.jpg

def calculate_prob(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))

    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

# Tinh xac suat cho moi thuoc tinh phan chia theo class
def calculate_class_prob(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.iteritems():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculate_prob(x, mean, stdev)

    return probabilities

# Du doan vector thuoc phan lop nao

def predict(summaries, inputVector):
    probabilities = calculate_class_prob(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.iteritems():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue

    return bestLabel

# Du doan tap du lieu testing thuoc vao phan lop nao

def get_predictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)

    return predictions

# Tinh toan do chinh xac cua phan lop

def get_accuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1

    return (correct / float(len(testSet))) * 100.0

def get_data_label(dataset):
    data = []
    label = []
    for x in dataset:
        data.append(x[:8])
        label.append(x[-1])

    return data, label

def main():
    filename = 'tieu_duong.csv'
    splitRatio = 0.8
    dataset = load_data(filename)
    trainingSet, testSet = split_data(dataset, splitRatio)

    print('Data size {0} \nTraining Size={1} \nTest Size={2}').format(len(dataset), len(trainingSet), len(testSet))

    # prepare model
    summaries = summarize_by_class(trainingSet)
    get_data_label(trainingSet)
    # test model
    predictions = get_predictions(summaries, testSet)
    accuracy = get_accuracy(testSet, predictions)
    print('Accuracy of my implement: {0}%').format(accuracy)

    # Compare with sklearn
    dataTrain, labelTrain = get_data_label(trainingSet)
    dataTest, labelTest = get_data_label(testSet)

    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(dataTrain, labelTrain)

    score = clf.score(dataTest, labelTest)

    print('Accuracy of sklearn: {0}%').format(score*100)


if __name__ == "__main__":
    main()
import sys
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class LogisticRegression:
    def __init__(self, fTrain, fValid, fTest, dictInput, trainOut, testOut, metricsOut, epochs):
        self.fTrain = fTrain
        self.fValid = fValid
        self.fTest = fTest
        self.dictInput = dictInput
        self.trainOut = trainOut
        self.testOut = testOut
        self.metricsOut = metricsOut
        self.epochs = epochs

        self.weights = None
        self.d = dict()
        self.dLen = None
        self.X = None
        self.y = None
        self.lr = 0.1

        self.logsTrain = np.array([])
        self.logsValid = np.array([])
        self.validX = None
        self.validY = None

    def readFile(self, file):
        f = open(file, "r")
        X = np.array([])
        y = np.array([], dtype=int)

        for line in f.readlines():
            featureSet = set()
            line = line.strip()
            lArr = np.array(line.split('\t'))
            y = np.append(y, int(lArr[0]))

            for feature in lArr[1:]:
                index, value = feature.split(':')
                featureSet.add(int(index))
                #featureDict[int(index)] = int(value)

            featureSet.add(self.dLen - 1)
            #featureDict[self.dLen - 1] = 1
            X = np.append(X, featureSet)

        f.close()
        return X, y


    @staticmethod
    def sparseDot(xi, weights):
        product = 0
        for i in xi:
            product +=  weights[i]
        return product


    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


    def SGD(self, xi, val):
        for i in xi:
            self.weights[i] += val

    def logLikelihood(self, X, y, logs):
        s = 0
        for index in range(X.shape[0]):
            xi = X[index]
            dotProduct = self.sparseDot(xi, self.weights)
            yi = y[index]
            s += -yi * dotProduct + np.log(1 + np.exp(dotProduct))
        s = s / X.shape[0]
        return np.append(logs, s)


    def fit(self):
        for i in range(self.epochs):
            for index in range(self.X.shape[0]):
                xi = self.X[index]
                dotProduct = self.sparseDot(xi, self.weights)

                p = self.sigmoid(dotProduct)

                self.SGD(xi, self.lr * (self.y[index] - p))
            self.logsTrain = self.logLikelihood(self.X, self.y, self.logsTrain)
            self.logsValid = self.logLikelihood(self.validX, self.validY, self.logsValid)


    def predict(self, X, y, labelsFilename):
        file = open(labelsFilename, "w+")
        wrong = 0
        for index, xi in enumerate(X):
            dotProduct = self.sparseDot(xi, self.weights)
            prediction = self.sigmoid(dotProduct)
            if prediction > 0.5:
                label = 1
            else:
                label = 0
            if label != y[index]: wrong += 1
            file.write("%s\n" % label)
        file.close()
        return wrong / X.shape[0]


    def createDict(self):
        readDict =  open(self.dictInput, "r")
        self.dLen = 0
        for line in readDict.readlines():
            word, index = line.split(" ")
            self.d[int(index)] = word
            self.dLen += 1
        self.weights = np.zeros(self.dLen)


    def writeError(self, trainError, testError):
        f = open(self.metricsOut, "w+")
        f.write("error(train): %s\n" %trainError)
        f.write("error(test): %s\n" %testError)
        f.close()

    def createPlots(self):
        df = pd.DataFrame({ "index": list(range(self.epochs)), "train": self.logsTrain, "valid": self.logsValid })
        plt.plot("index", "train", data=df)
        plt.plot("index", "valid", data=df)
        plt.legend()
        plt.show()

        #plot = sns.lineplot(x="index", y="valid", data=df)
        #plot.savefig("neglog.png")



if __name__ == "__main__":
    start = time.time()

    fTrain = sys.argv[1]
    fValid = sys.argv[2]
    fTest = sys.argv[3]
    dictInput = sys.argv[4]
    trainOut = sys.argv[5]
    testOut = sys.argv[6]
    metricsOut = sys.argv[7]
    epochs = int(sys.argv[8])
    lr = LogisticRegression(fTrain, fValid, fTest, dictInput, trainOut, testOut, metricsOut, epochs)
    print(str(time.time() - start) + " done intialization")

    lr.createDict()
    #print(str(time.time() - start) + " done creating dict")

    lr.X, lr.y = lr.readFile(lr.fTrain)
    lr.validX, lr.validY = lr.readFile(lr.fValid)
    #print(str(time.time() - start) + " done reading train file")

    lr.fit()
    #print(str(time.time() - start) + " done fitting")

    trainError = lr.predict(lr.X, lr.y, lr.trainOut)
    #print(str(time.time() - start) + " done predicting train")

    lr.testX, lr.testY = lr.readFile(lr.fTest)
    #print(str(time.time() - start) + " done reading test file")

    testError = lr.predict(lr.testX, lr.testY, lr.testOut)
    #print(str(time.time() - start) + " done predicting test")

    lr.writeError(trainError, testError)
   # print(str(time.time() - start) + " done writing error")

    #lr.createPlots()
    # end = time.time()
    # print(end - start)


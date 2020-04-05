import sys
import numpy as np


class DecisionStump:
    def __init__(self, train, test, index, trainLabels, testLabels, metrics):
        # Command line args
        self.train = train
        self.test = test
        self.index = index
        self.trainLabels = trainLabels
        self.testLabels = testLabels
        self.metrics = metrics
        # Data & Labels
        self.trainData = []
        self.testData = []
        self.left = ""
        self.leftLabel = ""
        self.rightLabel = ""

    def getDataFromTSV(self):
        self.trainData = np.genfromtxt(train, delimiter='\t',  skip_header=True, usecols=(index, -1), dtype=str)
        self.left = self.trainData[0][0]
        self.testData = np.genfromtxt(test, delimiter='\t', skip_header=True, usecols=(index, -1), dtype=str)

    def training(self):
        # First value train
        leftArr = self.trainData[self.trainData[:, 0] == self.left]
        uniqueLeft, countLeft = np.unique(leftArr[:, 1], return_counts=True)
        self.leftLabel = self.getMajorityLabel(uniqueLeft, countLeft)
        # Second value train
        rightArr = self.trainData[self.trainData[:, 0] != self.left]
        uniqueRight, countRight = np.unique(rightArr[:, 1], return_counts=True)
        self.rightLabel = self.getMajorityLabel(uniqueRight, countRight)


    @staticmethod
    def getMajorityLabel(unique, count):
        if len(unique) == 1: return unique[0]
        if count[0] > count[1]:
            return unique[0]
        else:
            return unique[1]

    def predictLabels(self, dataToLabels, labelsFile):
        f = open(labelsFile, "w+")
        numRows = dataToLabels.shape[0]
        numErrors = 0
        for i in range(numRows):
            row = dataToLabels[i]
            val = row[0]
            realLabel = row[1]
            if val == self.left:
                predictLabel = self.leftLabel
            else:
                predictLabel = self.rightLabel

            if predictLabel != realLabel:
                numErrors += 1

            f.write("%s\n" % predictLabel)
        f.close()
        return numErrors / numRows

    @staticmethod
    def writeErrorsFile(trainErrorRate, testErrorRate, metrics):
        f = open(metrics, "w+")
        f.write("error(train): %s\n" %trainErrorRate)
        f.write("error(test): %s" %testErrorRate)
        f.close()


def runDecisionStump(train, test, index, trainLabels, testLabels, metrics):
    stump = DecisionStump(train, test, index, trainLabels, testLabels, metrics)
    stump.getDataFromTSV()
    stump.training()
    trainErrorRate = stump.predictLabels(stump.trainData, trainLabels)
    testErrorRate = stump.predictLabels(stump.testData, testLabels)
    stump.writeErrorsFile(trainErrorRate, testErrorRate, metrics)


if __name__ == "__main__":
    # Command line arguments
    train = sys.argv[1]
    test = sys.argv[2]
    index = int(sys.argv[3])
    trainLabels = sys.argv[4]
    testLabels = sys.argv[5]
    metrics = sys.argv[6]
    # Run the decision stump algorithm
    runDecisionStump(train, test, index, trainLabels, testLabels, metrics)

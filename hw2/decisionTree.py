import sys
import numpy as np


class Leaf:
    def __init__(self, data):
        self.data = data
        self.label = self.getLabel()

    def getLabel(self):
        unique, count = np.unique(self.data[:, -1], return_counts=True)
        if len(unique) <= 0:
            return None
        elif len(unique) == 1:
            return unique[0]
        elif count[0] == count[1]:
            return max(unique[0], unique[1])
        else:
            i = count.argmax()
            return unique[i]


class Split:
    def __init__(self, feature, otherFeature, col, left, right, gain):
        self.feature = feature
        self.otherFeature = otherFeature
        self.col = col
        self.left = left
        self.right = right
        self.gain = gain


class Node:
    def __init__(self, split, leftSide, rightSide):
        self.split = split
        self.leftSide = leftSide
        self.rightSide = rightSide
        self.leftUnique, self.leftCount = np.unique(split.left[:, -1], return_counts=True)
        self.rightUnique, self.rightCount = np.unique(split.right[:, -1], return_counts=True)


class DecisionTree:
    def __init__(self, train, test, maxDepth):
        self.train = train
        self.test = test
        self.maxDepth = maxDepth
        # Calculated information
        self.trainData = None
        self.attributes = None
        self.testData = None
        self.root = None
        self.features = set()
        self.labels = set()

    def importFiles(self):
        data = np.genfromtxt(self.train, delimiter='\t', dtype=str)
        self.attributes = data[0, :]
        self.trainData = data[1:, :]
        self.testData = np.genfromtxt(self.test, delimiter='\t', skip_header=True, dtype=str)

    def getFeatures(self):
        uniqueFeatures = np.unique(self.trainData[:, 0:-1])
        uniqueLabels = np.unique(self.trainData[:, -1])
        self.features = set(uniqueFeatures)
        self.labels = set(uniqueLabels)

    def findOtherFeature(self, feature):
        return min(self.features.difference({feature}))

    def gini(self, data):
        unique, count = np.unique(data[:, -1], return_counts=True)
        gini = 0
        total = data.shape[0]
        for val in count:
            gini += (val / total) * (1 - (val / total))
        return gini

    def calculateGain(self, nodeGini, leftRows, rightRows):
        leftWeight = leftRows.shape[0] / (leftRows.shape[0] + rightRows.shape[0])
        rightWeight = 1 - leftWeight
        leftGini = self.gini(leftRows)
        rightGini = self.gini(rightRows)
        return nodeGini - (leftWeight * leftGini) - (rightWeight * rightGini)

    def splitData(self, data, col, feature):
        left = data[data[:, col] == feature]
        right = data[data[:, col] != feature]
        return left, right

    def findSplit(self, data):
        bestSplit = Split("", "", None, None, None, 0)
        initialGini = self.gini(data)
        numCols = data.shape[1] - 1
        for i in range(numCols):
            unique, count = np.unique(data[:, i], return_counts=True)
            if len(unique) == 0 or unique[0] == "None": continue
            for index, feature in enumerate(unique):
                left, right = self.splitData(data, i, feature)
                if left.shape[0] != 0 and right.shape[0] != 0:
                    gain = self.calculateGain(initialGini, left, right)
                    if gain > bestSplit.gain or gain == bestSplit.gain and max(feature, bestSplit.feature) == feature:
                        bestSplit = Split(feature, self.findOtherFeature(feature), i, left, right, gain)
        return bestSplit

    def build(self, data, depth):
        split = self.findSplit(data)
        if split.gain > 0 and depth < self.maxDepth:
            split.left[:, split.col] = None
            split.right[:, split.col] = None
            leftSide = self.build(split.left, depth + 1)
            rightSide = self.build(split.right, depth + 1)
            return Node(split, leftSide, rightSide)
        else:
            return Leaf(data)

    def printNode(self, node, direction):
        if direction == "left":
            unique, count = node.leftUnique, node.leftCount
        else:
            unique, count = node.rightUnique, node.rightCount
        result = []
        for label in self.labels:
            if label in unique:
                result.append(label)
                result.append(count[np.where(unique == label)])
            else:
                result.append(label)
                result.append(0)
        return result

    def printTree(self, node, depth=1):
        if isinstance(node, Leaf):
            pass
        else:
            leftValues = self.printNode(node, "left")
            print(depth * "|  " + "%s = %s [%d %s/%d %s]" % (
                self.attributes[node.split.col], node.split.feature, leftValues.pop(), leftValues.pop(), leftValues.pop(), leftValues.pop()))
            self.printTree(node.leftSide, depth + 1)

            rightValues = self.printNode(node, "right")
            print(depth * "|  " + "%s = %s [%d %s/%d %s]" % (
                self.attributes[node.split.col], node.split.otherFeature, rightValues.pop(), rightValues.pop(), rightValues.pop(), rightValues.pop()))
            self.printTree(node.rightSide, depth + 1)

    def classify(self, row, node):
        if isinstance(node, Leaf):
            return node.label
        else:
            if row[node.split.col] == node.split.feature:
                return self.classify(row, node.leftSide)
            else:
                return self.classify(row, node.rightSide)

    def writeLabelsAndCalculateError(self, data, out):
        f = open(out, "w+")
        total = data.shape[0]
        numErrors = 0
        for row in data:
            label = self.classify(row, self.root)
            if label != row[-1]:
                numErrors += 1
            f.write("%s\n" % label)
        f.close()
        return numErrors/total

    def writeError(self, trainError, testError, metricsOut):
        f = open(metricsOut, "w+")
        f.write("error(train): %s\n" % trainError)
        f.write("error(test): %s" % testError)
        f.close()


def runDecisionTree(train, test, maxDepth, trainOut, testOut, metricsOut):
    # Instantiate the tree
    model = DecisionTree(train, test, maxDepth)
    model.importFiles()
    model.getFeatures()
    # Build the model
    model.root = model.build(model.trainData, 0)
    # Print the model
    unique, count = np.unique(model.trainData[:, -1], return_counts=True)
    print("[%d %s/%d %s]" % (count[0], unique[0], count[1], unique[1]))
    model.printTree(model.root)
    # test the training and testing data, write out the errors and labels
    trainError = model.writeLabelsAndCalculateError(model.trainData, trainOut)
    testError = model.writeLabelsAndCalculateError(model.testData, testOut)
    model.writeError(trainError, testError, metricsOut)

    return trainError, testError


if __name__ == "__main__":
    train = sys.argv[1]
    test = sys.argv[2]
    maxDepth = int(sys.argv[3])
    trainOut = sys.argv[4]
    testOut = sys.argv[5]
    metricsOut = sys.argv[6]
    runDecisionTree(train, test, maxDepth, trainOut, testOut, metricsOut)



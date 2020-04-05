import sys
import numpy as np


class Inspect:
    def __init__(self, train, output):
        self.train = train
        self.output = output
        self.trainData = None
        self.error = None
        self.gini = None

    def getDataFromTSV(self):
        self.trainData = np.genfromtxt(self.train, delimiter='\t', skip_header=True, dtype=str)

    def findGini(self):
        unique, count = np.unique(self.trainData[:, -1], return_counts=True)
        gini = 0
        total = self.trainData.shape[0]
        max = 0
        for val in count:
            if (val > max):
                max = val
            gini += (val/total) * (1 - (val/total))
        error = 1 - (max / total)
        self.error = error
        self.gini = gini

    def writeToFile(self):
        f = open(self.output, "w+")
        f.write("gini_impurity: %s\n" %self.gini)
        f.write("error: %s\n" %self.error)
        f.close()


if __name__ == "__main__":
    train = sys.argv[1]
    output = sys.argv[2]
    inspect = Inspect(train, output)
    inspect.getDataFromTSV()
    inspect.findGini()
    inspect.writeToFile()


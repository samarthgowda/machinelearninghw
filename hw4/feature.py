import sys

class Feature:
    def __init__(self, trainInput, validationInput, testInput, dictInput, formattedTrainOut, formattedValidationOut, formattedTestOut, featureFlag):
        self.trainInput = trainInput
        self.validationInput = validationInput
        self.testInput = testInput
        self.dictInput = dictInput
        self.formattedTrainOut = formattedTrainOut
        self.formattedValidationOut = formattedValidationOut
        self.formattedTestOut = formattedTestOut
        self.featureFlag = featureFlag
        self.dict = dict()
        self.trimmingThreshold = 4


    def parse(self, file, fileOut):
        out = open(fileOut, "w+")
        file = open(file, "r")
        for line in file.readlines():
            seen = dict()
            label, text = line.split('\t', 1)
            textArr = text.split(' ')
            out.write(label)
            for word in textArr:
                if word in self.dict:
                    seen[word] = seen.get(word, 0) + 1

            for word, count in seen.items():
                if self.featureFlag == 2:
                    if count < self.trimmingThreshold:
                        out.write('\t' + str(self.dict[word]) + ":1")
                else:
                    out.write('\t' + str(self.dict[word]) + ":1")

            out.write('\n')
        file.close()
        out.close()


    def createDict(self):
        file = open(self.dictInput, "r")
        for line in file.readlines():
            word, index = line.split(" ")
            self.dict[word] = int(index)





if __name__ == "__main__":
    # Inputs
    trainInput = sys.argv[1]
    validationInput = sys.argv[2]
    testInput = sys.argv[3]
    dictInput = sys.argv[4]
    # Outputs
    formattedTrainOut = sys.argv[5]
    formattedValidationOut = sys.argv[6]
    formattedTestOut = sys.argv[7]
    # Feature Flag
    featureFlag = int(sys.argv[8])

    f = Feature(trainInput, validationInput, testInput, dictInput, formattedTrainOut, formattedValidationOut, formattedTestOut, featureFlag)
    f.createDict()
    f.parse(f.trainInput, f.formattedTrainOut)
    f.parse(f.validationInput, f.formattedValidationOut)
    f.parse(f.testInput, f.formattedTestOut)

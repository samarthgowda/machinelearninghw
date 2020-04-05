from decisionTree import runDecisionTree

trainError = []
testError = []
for i in range(0, 8):
    train, test = runDecisionTree("./handout/politicians_train.tsv",
                    "./handout/politicians_test.tsv",
                    i, "train.labels",
                    "test.labels",
                    "metrics.txt")
    trainError.append(train)
    testError.append(test)
print(trainError)
print(testError)

import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(b):
    exp = np.exp(b)
    s = np.sum(exp)
    return exp / s

class NeuralNet:
    def __init__(self, train_input, test_input, train_out, test_out, metrics_out, num_epoch, hidden_units, init_flag, lr):
        self.train_input = train_input
        self.test_input = test_input
        self.train_out = open(train_out, "w+")
        self.test_out = open(test_out, "w+")
        self.metrics_out = open(metrics_out, "w+")
        self.num_epoch = num_epoch
        self.hidden_units = hidden_units
        self.init_flag = init_flag
        self.lr = lr

        self.alpha = None
        self.beta = None
        self.train_X = None
        self.train_y = None
        self.test_X = None
        self.test_y = None
        self.num_classes = 10

        self.entropy_train = []
        self.entropy_test = []

    def init_weights(self):
        n_rows, n_cols = self.train_X.shape
        if self.init_flag == 1:
            self.alpha = np.random.uniform(-0.1, 0.1, size=(self.hidden_units, n_cols))
            self.beta = np.random.uniform(-0.1, 0.1, size=(self.num_classes, self.hidden_units + 1))
        elif self.init_flag == 2:
            self.alpha = np.zeros((self.hidden_units, n_cols))
            self.beta = np.zeros((self.num_classes, self.hidden_units + 1))
        else:
            raise Exception("init_flag is not valid")

    def read_files(self):
        train = np.genfromtxt(self.train_input, delimiter=",", dtype=float)
        self.train_X = train[:, 1:]
        self.train_y = train[:, 0]
        test = np.genfromtxt(self.test_input, delimiter=",", dtype=float)
        self.test_X = test[:, 1:]
        self.test_y = test[:, 0]

        n, m = self.train_X.shape
        train_X0 = np.ones((n, 1))
        self.train_X = np.hstack((train_X0, self.train_X))

        n, m = self.test_X.shape
        test_X0 = np.ones((n, 1))
        self.test_X = np.hstack((test_X0, self.test_X))

        # self.num_classes = len(np.unique(self.train_y))

    def train(self):
        y = self.train_y.astype(int)
        for i in range(self.num_epoch):
            for j, xi in enumerate(self.train_X):
                a = xi.dot(self.alpha.T)
                z = sigmoid(a)
                z_ = np.append([1], z)
                b = z_.dot(self.beta.T)
                y_prime = softmax(b)
                # end of forward
                db = np.copy(y_prime)
                db[y[j]] -= 1

                dbeta = np.asmatrix(db).T.dot(np.asmatrix(z_))

                dz = db.T.dot(self.beta[:, 1:]).T
                da = dz * z * (1-z)

                dalpha = np.asmatrix(da).T.dot(np.asmatrix(xi))
                # end of backprop
                self.beta += - self.lr * dbeta
                self.alpha += - self.lr * dalpha

            self.mean_cross_entropy(self.train_X, self.train_y, i + 1, "train")
            self.mean_cross_entropy(self.test_X, self.test_y, i + 1, "test")


    def mean_cross_entropy(self, X, y, epoch, dataset):
        y = y.astype(int)
        J = 0
        for j, xi in enumerate(X):
            a = xi.dot(self.alpha.T)
            z = sigmoid(a)
            z_ = np.append([1], z)
            b = z_.dot(self.beta.T)
            y_prime = softmax(b)
            J += np.log(y_prime[y[j]])
        mean_cross_entropy = -(1/X.shape[0]) * J
        if dataset is not None: self.metrics_out.write("epoch={0} crossentropy({1}): {2}\n".format(epoch, dataset, mean_cross_entropy))
        if dataset == "train": self.entropy_train.append(mean_cross_entropy)
        if dataset == "test": self.entropy_test.append(mean_cross_entropy)
        # self.avg_cross_entropy[dataset] = self.avg_cross_entropy[dataset].append(mean_cross_entropy)
        return mean_cross_entropy

    def predict(self, X, y, dataset, labels_file):
        num_errors = 0
        y = y.astype(int)
        for j, xi in enumerate(X):
            a = xi.dot(self.alpha.T)
            z = sigmoid(a)
            z_ = np.append([1], z)
            b = z_.dot(self.beta.T)
            y_prime = softmax(b)
            y_ki = y_prime.argmax()
            if y_ki != y[j]: num_errors += 1
            labels_file.write("{}\n".format(y_ki))

        self.metrics_out.write("error({0}): {1}\n".format(dataset, num_errors / X.shape[0]))


if __name__ == "__main__":

    train_input = sys.argv[1]
    test_input = sys.argv[2]
    train_out = sys.argv[3]
    test_out = sys.argv[4]
    metrics_out = sys.argv[5]
    num_epoch = int(sys.argv[6])
    hidden_units = int(sys.argv[7])
    init_flag = int(sys.argv[8])
    lr = float(sys.argv[9])

    train_entropy = []
    test_entropy = []
    # h_units = [5, 20, 50, 100, 200]
    # h_units = [5, 10, 20, 30]
    # lrs = [0.1, 0.01, 0.001]
    # for h_unit in h_units:
    nn = NeuralNet(train_input, test_input, train_out, test_out, metrics_out, num_epoch, hidden_units, init_flag, lr)
    print("running neural network on {0} with {1} epochs, {2} hidden units".format(nn.train_input, nn.num_epoch, nn.hidden_units))
    nn.read_files()
    nn.init_weights()
    nn.train()
    # train_entropy.append(nn.mean_cross_entropy(nn.train_X, nn.train_y, 1, None))
    # test_entropy.append(nn.mean_cross_entropy(nn.test_X, nn.test_y, 1, None))
    df = pd.DataFrame({"epochs": list(range(0, nn.num_epoch)), "train": nn.entropy_train, "test": nn.entropy_test})
    plt.plot("epochs", "train", data=df)
    plt.plot("epochs", "test", data=df)
    plt.title("Average cross-entropy vs number of epochs for {0} learning rate".format(nn.lr))
    plt.ylim(0, 1)
    plt.legend()
    plt.show()
    # nn.predict(nn.train_X, nn.train_y, "train", nn.train_out)
    # nn.predict(nn.test_X, nn.test_y, "test", nn.test_out)

    nn.train_out.close()
    nn.test_out.close()
    nn.metrics_out.close()

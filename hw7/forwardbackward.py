import numpy as np
import sys
import warnings
class FB:
    def __init__(self, test_input, index_to_word, index_to_tag, hmmprior, hmmemit, hmmtrans, predicted_file, metric_file):
        self.test_input = open(test_input, "r")
        self.index_to_word_file = index_to_word
        self.index_to_tag_file = index_to_tag
        self.pi = np.loadtxt(hmmprior, dtype=float)
        self.emission = np.loadtxt(hmmemit, dtype=float)
        self.transition = np.loadtxt(hmmtrans, dtype=float)
        self.predicted_file = open(predicted_file, "w+")
        self.metric_file = metric_file

        self.index_to_tag = None
        self.index_to_word = None
        self.word_to_index = None
        self.tag_to_index = None
        self.test = None

        self.log_likelihood = 0
        self.num_errors = 0
        self.total_predicted = 0
        self.num_seq = 0
        self.num_warnings = 0

    @staticmethod
    def parse_index_file(file):
        file_array = np.genfromtxt(file, delimiter="\n", dtype=str)
        file_dict = dict()
        file_dict_reversed = dict()
        for index, val in enumerate(file_array):
            file_dict[val] = index
            file_dict_reversed[index] = val
        return file_dict, file_dict_reversed

    def forward_backward(self, seq):
        seq_split = seq.split(" ")
        T = len(seq_split)
        alpha = np.zeros(shape=(self.pi.shape[0], self.emission.shape[1]))
        beta = np.zeros(shape=(self.pi.shape[0], self.emission.shape[1]))
        for index, word_tag in enumerate(seq_split):
            word, tag = word_tag.strip().split("_")
            word_i = self.word_to_index[word]

            if index == 0:
                a = np.multiply(self.emission[:, word_i], self.pi)
            else:
                dot = np.dot(self.transition.T, alpha[:, index - 1])
                dot_max = np.max(dot)
                a = np.multiply(self.emission[:, word_i], dot_max + np.log(np.exp(dot - dot_max)))


            alpha[: , index] = a

        for index, word_tag in enumerate(reversed(seq_split)):
            word, tag = word_tag.strip().split("_")
            word_i = self.word_to_index[word]
            if index == 0:
                beta[:,  T - 1] = 1
            else:
                multiply = np.multiply(self.emission[:, prev_word_i], beta[:, T - index])
                mult_max = np.max(multiply)
                b = mult_max + np.log(np.exp(np.dot(self.transition, multiply) - mult_max))
                beta[:, T - index - 1] = b
            prev_word, prev_tag, prev_word_i = word, tag, word_i

        self.predict_metrics(seq_split, alpha, beta)

    def predict_metrics(self, seq_split, alpha, beta):
        T = len(seq_split)
        # if type(np.sum(alpha[:, T-1])) != np.float64:
        #     print("not float")

        # with warnings.catch_warnings():
        #     warnings.filterwarnings('error')
        #     try:
        #         random_val = np.log(np.sum(alpha[:, T - 1]))
        #     except Warning:
        #         if self.num_warnings == 0:
        #             print("T: {0}   alpha".format(T))
        #             print(alpha[:, T-1])
        #             print("\n")
        #         self.num_warnings += 1


        #print(np.sum(alpha[:, T-1]))
        self.log_likelihood += np.sum(alpha[:, T - 1])
        prediction_matrix = np.multiply(alpha, beta)
        argmax = np.argmax(prediction_matrix, axis = 0)

        for index, word_tag in enumerate(seq_split):
            word, tag = word_tag.strip().split("_")
            tag_i = self.tag_to_index[tag]

            self.total_predicted += 1
            if argmax[index] != tag_i:
                self.num_errors += 1
            space = (" " if index < T - 1 else "")
            self.predicted_file.write(("{0}_{1}" + space).format(word, self.index_to_tag[argmax[index]]))

        self.predicted_file.write("\n")

    def write_metrics(self):
        f_metric = open(self.metric_file, "w+")
        f_metric.write("Average Log-Likelihood: {0}\n".format(self.log_likelihood / self.num_seq))
        f_metric.write("Accuracy: {0}".format(1 - (self.num_errors / self.total_predicted)))


def run_fb(test_input, index_to_word, index_to_tag, hmmprior, hmmemit, hmmtrans, predicted_file, metric_file):
    fb = FB(test_input, index_to_word, index_to_tag, hmmprior, hmmemit, hmmtrans, predicted_file, metric_file)
    fb.word_to_index, fb.index_to_word = fb.parse_index_file(fb.index_to_word_file)
    fb.tag_to_index, fb.index_to_tag = fb.parse_index_file(fb.index_to_tag_file)

    for seq in fb.test_input.readlines():
        fb.forward_backward(seq)
        fb.num_seq += 1

    fb.write_metrics()
    fb.predicted_file.close()

    return fb.log_likelihood / fb.num_seq


if __name__ == "__main__":
    test_input = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmmprior = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]
    predicted_file = sys.argv[7]
    metric_file = sys.argv[8]

    run_fb(test_input, index_to_word, index_to_tag, hmmprior, hmmemit, hmmtrans, predicted_file, metric_file)
import numpy as np
import sys

class HMM:
    def __init__(self, train_input, index_to_word, index_to_tag, hmmprior, hmmemit, hmmtrans):
        self.train_input_file = train_input
        self.index_to_word_file = index_to_word
        self.index_to_tag_file = index_to_tag
        self.prior_output = hmmprior
        self.emit_output = hmmemit
        self.trans_output = hmmtrans

        self.index_to_tag = None
        self.index_to_word = None
        self.transition = None
        self.emission = None
        self.pi = None
        self.train = None
        self.num_seq = None

    @staticmethod
    def parse_index_file(file):
        file_array = np.genfromtxt(file, delimiter="\n", dtype=str)
        file_dict = dict()
        for index, val in enumerate(file_array):
            file_dict[val] = index
        return file_dict

    @staticmethod
    def read_data_file(file):
        return np.genfromtxt(file, delimiter="\n", dtype=str)

    def parse_train(self):
        for current_seq_index, seq in enumerate(self.train):
            if self.num_seq is None or current_seq_index < self.num_seq:
                prev_tag = None
                word_count = 0
                for word_tag in seq.split(" "):
                    word, tag = word_tag.split("_")

                    ''' Error handling '''
                    if word not in self.index_to_word:
                        raise Exception("Word not in index to word")
                    if tag not in self.index_to_tag:
                        raise Exception("Tag not in index to tag")

                    ''' Adding to the emission matrix '''
                    self.emission[self.index_to_tag[tag], self.index_to_word[word]] += 1

                    ''' Adding to the initial pi vector '''
                    if word_count == 0:
                        self.pi[self.index_to_tag[tag]] += 1
                    word_count += 1

                    ''' Adding to the transition matrix '''
                    if prev_tag is not None:
                        self.transition[self.index_to_tag[prev_tag], self.index_to_tag[tag]] += 1

                    prev_tag = tag

        ''' Pseudocount and calculate probabilities '''
        self.pi /= np.sum(self.pi)

        self.transition /= self.transition.sum(axis = 1, keepdims = True)

        self.emission /= self.emission.sum(axis = 1, keepdims = True)

    @staticmethod
    def write_outputs(file, arr):
        f = open(file, "wb")
        np.savetxt(f, arr, delimiter=" ", newline = "\n")
        f.close()

def run_learnhmm(train_input, index_to_word, index_to_tag, hmmprior, hmmemit, hmmtrans, num_seq = None):
    hmm = HMM(train_input, index_to_word, index_to_tag, hmmprior, hmmemit, hmmtrans)
    hmm.index_to_word = hmm.parse_index_file(hmm.index_to_word_file)
    hmm.index_to_tag = hmm.parse_index_file(hmm.index_to_tag_file)

    hmm.transition = np.ones(shape=(len(hmm.index_to_tag), len(hmm.index_to_tag)))
    hmm.emission = np.ones(shape=(len(hmm.index_to_tag), len(hmm.index_to_word)))
    hmm.pi = np.ones(len(hmm.index_to_tag))

    hmm.train = hmm.read_data_file(hmm.train_input_file)
    if num_seq is not None:
        hmm.num_seq = num_seq
    hmm.parse_train()

    hmm.write_outputs(hmm.prior_output, hmm.pi)
    hmm.write_outputs(hmm.trans_output, hmm.transition)
    hmm.write_outputs(hmm.emit_output, hmm.emission)


if __name__ == "__main__":
    train_input = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmmprior = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]

    run_learnhmm(train_input, index_to_word, index_to_tag, hmmprior, hmmemit, hmmtrans)

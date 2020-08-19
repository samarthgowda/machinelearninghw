import sys
from learnhmm import run_learnhmm
from forwardbackward import run_fb
import pandas as pd
import matplotlib.pyplot as plt


df = pd.DataFrame(columns=["sequences", "train", "test"])
for num_seq in (10, 100, 1000, 10000):
    run_learnhmm("./handout/trainwords.txt",
                 "./handout/index_to_word.txt",
                 "./handout/index_to_tag.txt",
                 "./hmmprior.txt",
                 "./hmmemit.txt",
                 "./hmmtrans.txt",
                 num_seq=num_seq)

    train_log = run_fb("./handout/trainwords.txt",
           "./handout/index_to_word.txt",
           "./handout/index_to_tag.txt",
           "./hmmprior.txt",
           "./hmmemit.txt",
           "./hmmtrans.txt",
           "./predicted.txt",
           "./metric.txt")
    print("train_log    {0}".format(train_log))

    test_log = run_fb("./handout/testwords.txt",
                       "./handout/index_to_word.txt",
                       "./handout/index_to_tag.txt",
                       "./hmmprior.txt",
                       "./hmmemit.txt",
                       "./hmmtrans.txt",
                       "./predicted.txt",
                       "./metric.txt")

    print("test_log    {0}".format(test_log))

    df = df.append(
        { 'sequences': num_seq,
        'train': train_log,
        'test': test_log },
        ignore_index=True
    )

plt.plot("sequences", "train", data=df)
plt.plot("sequences", "test", data=df)
plt.title("Average log-likelihood for various sequences on train, test data")
plt.legend()
plt.show()



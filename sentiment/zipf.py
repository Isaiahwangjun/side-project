from pylab import *
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('term_freq_df.csv', index_col=0)
counts = df.total
tokens = df.index
ranks = arange(1, len(counts) + 1)
indices = argsort(-counts)
frequencies = counts[indices]
plt.figure(figsize=(8, 6))
plt.ylim(1, 10**6)
plt.xlim(1, 10**6)
loglog(ranks, frequencies, marker=".")
plt.plot([1, frequencies[0]], [frequencies[0], 1], color='r')
title("Zipf plot for tweets tokens")
xlabel("Frequency rank of token")
ylabel("Absolute frequency of token")
grid(True)
for n in list(logspace(-0.5, log10(len(counts) - 2), 25).astype(int)):
    dummy = text(ranks[n],
                 frequencies[n],
                 " " + tokens[indices[n]],
                 verticalalignment="bottom",
                 horizontalalignment="left")
plt.show()

#https://towardsdatascience.com/another-twitter-sentiment-analysis-bb5b01ebad90

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

cols = ['sentiment', 'id', 'date', 'query_string', 'user', 'text']
df = pd.read_csv("train.csv", header=None, names=cols, encoding='ISO-8859-1')

# delete unnecessary features
df.drop(['id', 'date', 'query_string', 'user'], axis=1, inplace=True)

# calculate and add the length of pre clean
df['pre_clean_len'] = [len(t) for t in df.text]


# next step need to clean the text
def tweet_cleaner(text):
    # remove HTMML decoding
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(text, 'html.parser')
    souped = soup.get_text()

    # remove tag and URL
    import re
    rm1 = r'@[A-Za-z0-9_]+'
    rm2 = r'https?://[^ ]+'
    combined_pat = r'|'.join((rm1, rm2))
    stripped = re.sub(combined_pat, '', souped)

    # all the word change to lower case
    lower_case = stripped.lower()

    # use 'tok' to divde the sentence to token (this step will remove additional space),
    # and use 'join' to add space between two tokens, so can get complete sentence
    from nltk.tokenize import WordPunctTokenizer
    tok = WordPunctTokenizer()
    words = tok.tokenize(lower_case)
    return (" ".join(words)).strip()


clean_tweet_texts = []
for i in range(len(df)):
    clean_tweet_texts.append(tweet_cleaner(df['text'][i]))

clean_df = pd.DataFrame(clean_tweet_texts, columns=['text'])

# column name : sentiment -> target
clean_df['target'] = df.sentiment

# label 4->1
clean_df.replace({'target': {4: 1}}, inplace=True)

clean_df.to_csv('clean_tweet.csv', encoding='utf-8')

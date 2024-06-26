import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('clean_tweet.csv', index_col=0)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
x = df.text
y = df.target

from sklearn.model_selection import train_test_split

x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(
    x, y, test_size=.02)
x_validation, x_test, y_validation, y_test = train_test_split(
    x_validation_and_test, y_validation_and_test, test_size=.5)
print(f"訓練集共有{len(x_train)} 個條目，\
    其中{ (len(x_train[y_train == 0]) / (len(x_train) * 1.)) * 100:.2f}% 為負，\
    {(len(x_train[y_train == 1]) / (len(x_train) * 1.)) * 100:.2f}% 為正")
print(f"驗證集共有{len(x_validation)} 個條目，\
    其中{ (len(x_validation[y_validation == 0]) / (len(x_validation) * 1.)) * 100:.2f}% 為負，\
    {(len(x_validation[y_validation == 1]) / (len(x_validation) * 1.)) * 100:.2f}% 為正"
      )
print(f"測試集共有{len(x_test)} 個條目，\
    其中{(len(x_test[y_test == 0]) / (len(x_test) * 1.)) * 100:.2f}% 為陰性，\
    {(len(x_test[y_test == 1]) / (len(x_test) * 1.)) * 100:.2f}% 為陽性")

tbresult = [TextBlob(i).sentiment.polarity for i in x_validation]
tbpred = [0 if n < 0 else 1 for n in tbresult]
conmat = np.array(confusion_matrix(y_validation, tbpred, labels=[1, 0]))
confusion = pd.DataFrame(conmat,
                         index=['positive', 'negative'],
                         columns=['predicted_positive', 'predicted_negative'])
print("Accuracy Score: {0:.2f}%".format(
    accuracy_score(y_validation, tbpred) * 100))
print("-" * 80)
print("Confusion Matrix\n")
print(confusion)
print("-" * 80)
print("Classification Report\n")
print(classification_report(y_validation, tbpred))

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from time import time


def accuracy_summary(pipeline, x_train, y_train, x_test, y_test):
    if len(x_test[y_test == 0]) / (len(x_test) * 1.) > 0.5:
        null_accuracy = len(x_test[y_test == 0]) / (len(x_test) * 1.)
    else:
        null_accuracy = 1. - (len(x_test[y_test == 0]) / (len(x_test) * 1.))
    t0 = time()
    sentiment_fit = pipeline.fit(x_train, y_train)
    y_pred = sentiment_fit.predict(x_test)
    train_test_time = time() - t0
    accuracy = accuracy_score(y_test, y_pred)
    print("null accuracy: {0:.2f}%".format(null_accuracy * 100))
    print("accuracy score: {0:.2f}%".format(accuracy * 100))
    if accuracy > null_accuracy:
        print("model is {0:.2f}% more accurate than null accuracy".format(
            (accuracy - null_accuracy) * 100))
    elif accuracy == null_accuracy:
        print("model has the same accuracy with the null accuracy")
    else:
        print("model is {0:.2f}% less accurate than null accuracy".format(
            (null_accuracy - accuracy) * 100))
    print("train and test time: {0:.2f}s".format(train_test_time))
    print("-" * 80)
    return accuracy, train_test_time


cvec = CountVectorizer()
lr = LogisticRegression()
n_features = np.arange(10000, 100001, 10000)


def nfeature_accuracy_checker(vectorizer=cvec,
                              n_features=n_features,
                              stop_words=None,
                              ngram_range=(1, 1),
                              classifier=lr):
    result = []
    print(classifier)
    print("\n")
    for n in n_features:
        vectorizer.set_params(stop_words=stop_words,
                              max_features=n,
                              ngram_range=ngram_range)
        checker_pipeline = Pipeline([('vectorizer', vectorizer),
                                     ('classifier', classifier)])
        print("Validation result for {} features".format(n))
        nfeature_accuracy, tt_time = accuracy_summary(checker_pipeline,
                                                      x_train, y_train,
                                                      x_validation,
                                                      y_validation)
        result.append((n, nfeature_accuracy, tt_time))
    return result


csv = 'term_freq_df.csv'
term_freq_df = pd.read_csv(csv, index_col=0)

from sklearn.feature_extraction import text

my_stop_words = frozenset(
    list(term_freq_df.sort_values(by='total', ascending=False).iloc[:8].index))
my_stop_words = list(my_stop_words)

print("RESULT FOR UNIGRAM WITHOUT STOP WORDS\n")
feature_result_wosw = nfeature_accuracy_checker(stop_words='english')
print("RESULT FOR UNIGRAM WITH STOP WORDS\n")
feature_result_ug = nfeature_accuracy_checker()
print("RESULT FOR UNIGRAM WITHOUT CUSTOM STOP WORDS (Top 10 frequent words)\n")
feature_result_wocsw = nfeature_accuracy_checker(stop_words=my_stop_words)

import matplotlib.pyplot as plt

nfeatures_plot_ug = pd.DataFrame(
    feature_result_ug,
    columns=['nfeatures', 'validation_accuracy', 'train_test_time'])
nfeatures_plot_ug_wocsw = pd.DataFrame(
    feature_result_wocsw,
    columns=['nfeatures', 'validation_accuracy', 'train_test_time'])
nfeatures_plot_ug_wosw = pd.DataFrame(
    feature_result_wosw,
    columns=['nfeatures', 'validation_accuracy', 'train_test_time'])
plt.figure(figsize=(8, 6))
plt.plot(nfeatures_plot_ug.nfeatures,
         nfeatures_plot_ug.validation_accuracy,
         label='with stop words')
plt.plot(nfeatures_plot_ug_wocsw.nfeatures,
         nfeatures_plot_ug_wocsw.validation_accuracy,
         label='without custom stop words')
plt.plot(nfeatures_plot_ug_wosw.nfeatures,
         nfeatures_plot_ug_wosw.validation_accuracy,
         label='without stop words')
plt.title("Without stop words VS With stop words (Unigram): Accuracy")
plt.xlabel("Number of features")
plt.ylabel("Validation set accuracy")
plt.legend()
plt.show()

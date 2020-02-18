import time
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from nltk.corpus import stopwords
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score, make_scorer

from sklearn.linear_model import LogisticRegression
from sklearn import svm

# You will have to download the set of stop words the first time
import nltk
# nltk.download('stopwords')
from sklearn.metrics import f1_score

# train train_partition samples
train_partition = 2911
n_classes = 4
label_dict = {'__label__0': 0, '__label__1': 1,
              '__label__2': 2, '__label__3': 3}

stop_words = stopwords.words('english')
stop_words.append("<eos>")
new_stopwords_list = set(stop_words)
stopset = list(new_stopwords_list)

data1 = pd.read_csv('train_frightening.csv', sep='\t')
# data2 = pd.read_csv('test_frightening.csv', sep='\t')
data3 = pd.read_csv('dev_frightening.csv', sep='\t')

train_data = data1
dev_data = data3

def text_to_wordlist_no_stop(raw_text):
    # read long string fom pandas dataframe
    # reconstruct as lists of sentences, sentence = list of strings
    # return raw_text.split()
    # stop_words = stopwords.words('english')
    tokenized_words = [x.lower() for x in raw_text.split()]
    wordlist_no_stop = [
        word for word in tokenized_words if word not in stopset]
    return ' '.join(wordlist_no_stop)


def list_to_string(text_list):
    # input list of words
    # output long string
    return ' '.join(text_list)

def map_vectorization_on_batch(list_of_text):
    return list(map(bert_text_encode, list_of_text))


# to access an element in df, df.iloc[row_index][col_label]
# print(i)
data_batch = train_data.append(dev_data)
label_batch = data_batch['label']
text_batch = data_batch['text']

# reconstruct each text into long list of words
# whole dataset text_list = num of article * one big word strings
text_list = list(map(text_to_wordlist_no_stop, text_batch))

# bag of words
# vectorizer = CountVectorizer()
# text_vec_list = vectorizer.fit_transform(text_list)
# # print(vectorizer.get_feature_names())
# X = text_vec_list.toarray()

# ### 2 gram
# vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2))
# text_vec_list_2gram = vectorizer2.fit_transform(text_list)
# # print(vectorizer2.get_feature_names())
# print(text_vec_list_2gram.toarray())
# X = text_vec_list.toarray()

# tf-idf
vectorizer_tf = TfidfVectorizer()
text_vec_list_tf = vectorizer_tf.fit_transform(text_list)
# print(text_vec_list_tf.toarray())
X = text_vec_list_tf.toarray()

# SVM classifier
print('SVM classifier')
clf = svm.SVC(kernel='linear', C=1)
clf.fit(X[0:train_partition], label_batch[0:train_partition])
y_pred = clf.predict(X[train_partition:])
print(classification_report(label_batch[train_partition:], y_pred))

# Logistic
# print('Logistic')
# clf = LogisticRegression(random_state=0)
# clf.fit(X[0:train_partition], label_batch[0:train_partition])
# y_pred = clf.predict(X[train_partition:])
# print(classification_report(label_batch[train_partition:], y_pred))


#
# dummy_clf = DummyClassifier(strategy="stratified")
# dummy_clf.fit(X[0:train_partition], label_batch[0:train_partition])
# y_pred = dummy_clf.predict(X[train_partition:])
# print(classification_report(label_batch[train_partition:], y_pred))

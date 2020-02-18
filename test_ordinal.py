import numpy as np
import torch
from torch import nn

from skorch import NeuralNet

from spacecutter.callbacks import AscensionCallback
from spacecutter.losses import CumulativeLinkLoss
from spacecutter.models import OrdinalLogisticModel

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

nltk.download('stopwords')
from sklearn.metrics import f1_score

# train 2911 samples
n_classes = 4
label_dict = {'__label__0': 0, '__label__1': 1, '__label__2': 2, '__label__3': 3}

'''Model Param'''
output_size = n_classes
stop_words = stopwords.words('english')
stop_words.append("<eos>")
new_stopwords_list = set(stop_words)
stopset = list(new_stopwords_list)

data1 = pd.read_csv('train_frightening.csv', sep='\t')
# data2 = pd.read_csv('test_frightening.csv', sep='\t')
data3 = pd.read_csv('dev_frightening.csv', sep='\t')

train_data = data1
dev_data = data3
print('dev:', len(dev_data.index))


def label_to_one_hot(label_list, n_classes):
	label_list_to_digits = [label_dict[each] for each in label_list]
	# seq lengh may vary because of last several samples
	y = torch.zeros(len(label_list_to_digits), n_classes)

	y[range(y.shape[0]), torch.LongTensor(label_list_to_digits)] = 1

	return y


def label_to_digits(label_list):
	label_list_to_digits = [label_dict[each] for each in label_list]

	return torch.LongTensor(label_list_to_digits)


def text_to_wordlist_no_stop(raw_text):
	# read long string fom pandas dataframe
	# reconstruct as lists of sentences, sentence = list of strings
	# return raw_text.split()
	# stop_words = stopwords.words('english')
	tokenized_words = [x.lower() for x in raw_text.split()]
	wordlist_no_stop = [word for word in tokenized_words if word not in stopset]
	return ' '.join(wordlist_no_stop)


def list_to_string(text_list):
	# input list of words
	# output long string
	return ' '.join(text_list)

# to access an element in df, df.iloc[row_index][col_label]
# data_batch = train_data.append(dev_data)

data_batch = train_data.iloc[0:1000]
partition = int(0.9 * len(data_batch.index))

print(partition)

label_batch = data_batch['label']
text_batch = data_batch['text']

# reconstruct each text into long list of words
# whole dataset text_list = num of article * one big word strings
text_list = list(map(text_to_wordlist_no_stop, text_batch))

label_batch_digits = label_to_one_hot(label_batch, n_classes)

y = label_batch_digits[0:partition]
y_true = label_batch_digits[partition:]

print(y)

### bag of words
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

### tf-idf
vectorizer_tf = TfidfVectorizer()
text_vec_list_tf = vectorizer_tf.fit_transform(text_list)
# print(text_vec_list_tf.toarray())
X = text_vec_list_tf.toarray()
print('X:', X)

### SVM classifier
# print('here!')
# clf = svm.SVC(kernel='linear', C=1)
# clf.fit(X[0:2911], label_batch[0:2911])
# y_pred = clf.predict(X[2911:])
# print(classification_report(label_batch[2911:], y_pred))

### Logistic
# print('here!')
# clf = LogisticRegression(random_state=0)
# clf.fit(X[0:2911], y)
# y_pred = clf.predict(X[2911:])
# print(classification_report(y_true, y_pred))

### Ordinal
num_features = X.shape[1]
num_classes = n_classes

print('n class:', num_classes)
print('num features:', X.shape[1])

predictor = nn.Sequential(
	nn.Linear(num_features, num_features),
	nn.ReLU(),
	nn.Linear(num_features, 1)
)

print('predictor:', predictor)

skorch_model = NeuralNet(
	module=OrdinalLogisticModel,
	module__predictor=predictor,
	module__num_classes=num_classes,
	criterion=CumulativeLinkLoss,
	max_epochs=20,
	train_split=None,
	callbacks=[
		('ascension', AscensionCallback()),
	],
)

skorch_model.fit(torch.from_numpy(X[0:partition]).float(), y.long())
print(skorch_model)

y_pred = skorch_model.predict(torch.from_numpy(X[partition:]).float())

print(classification_report(np.argmax(y_true.numpy(), axis=1), np.argmax(y_pred, axis=1)))

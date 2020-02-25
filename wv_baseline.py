import time
import pickle

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from nltk.corpus import stopwords
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score, make_scorer

from sklearn.linear_model import LogisticRegression
from sklearn import svm

import gensim.downloader as api

import torch
from torch import nn
import torch.nn.functional as F

from skorch import NeuralNetClassifier
from textrank import TextRank4Keyword

from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

textranker = TextRank4Keyword()

if torch.cuda.is_available():
	print("WARNING: You have a CUDA device")

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

cur_time = time.time()
glove_model = api.load("glove-wiki-gigaword-300")
glove_dim = 300
print('load word emb takes time:', time.time() - cur_time)

# You will have to download the set of stop words the first time
import nltk
# nltk.download('stopwords')
from sklearn.metrics import f1_score

# all 3639 samples
# train 2911 samples
text_piece = 100  # if change, also modify in chunks()
batch_size = 50
n_classes = 4
label_dict = {'__label__0': 0, '__label__1': 1, '__label__2': 2, '__label__3': 3}

epochs = 6
log_interval = 1
lr = 0.05
clip = 0.25
keyword_num = 200

'''Model Param'''
output_size = n_classes
embedding_dim = 768
hidden_dim = 768
n_layers = 1
stop_words = stopwords.words('english')
# stop_words.append("<eos>")
stop_words.extend(["<eos>", ".", ",", '.', "s", 's', 'b', 'd'])
new_stopwords_list = set(stop_words)
stopset = list(new_stopwords_list)

# print(stopset)

# data1 = pd.read_csv('train_frightening.csv', sep='\t')
# # data2 = pd.read_csv('test_frightening.csv', sep='\t')
# data3 = pd.read_csv('dev_frightening.csv', sep='\t')

# train_data = data1
# dev_data = data3
# all_data = pd.read_csv('severity/text_frightening_severity.csv', sep='\t')

all_files = ['severity/text_frightening_severity.csv',
			 'severity/text_alcohol_severity.csv',
			 'severity/text_nudity_severity.csv',
			 'severity/text_violence_severity.csv',
			 'severity/text_profanity_severity.csv']

f_baseline_filename = "all_file_NN_2_baseline_record.txt"


def label_to_one_hot(label_list, n_classes):
	label_list_to_digits = [label_dict[each] for each in label_list]
	# seq lengh may vary because of last several samples
	y = torch.zeros(len(label_list_to_digits), n_classes)

	y[range(y.shape[0]), torch.LongTensor(label_list_to_digits)] = 1

	return y


def label_to_digits(label_list):
	label_list_to_digits = [label_dict[each] for each in label_list]

	return torch.LongTensor(label_list_to_digits)


def combine_as_long_string(lst):
	return ' '.join(lst)


def reconstruct_text(raw_text):
	# read long string fom pandas dataframe
	# reconstruct as lists of sentences, sentence = list of strings
	# each text element is a long list of char, need to .split() to strings
	text = raw_text.split()
	# print(text)
	list_of_sent = []
	sent_container = []
	for each_word in text:
		if each_word != '<EOS>':
			sent_container.append(each_word)
		else:
			sent_container.append(each_word)
			list_of_sent.append(combine_as_long_string(sent_container))
			sent_container = []

	return list_of_sent


def text_to_wordlist(raw_text):
	# read long string fom pandas dataframe
	# reconstruct as lists of sentences, sentence = list of strings
	# return raw_text.split()
	return raw_text.split()


def text_to_wordlist_no_stop(raw_text):
	# read long string fom pandas dataframe
	# reconstruct as lists of sentences, sentence = list of strings
	# return raw_text long string
	# stop_words = stopwords.words('english')
	tokenized_words = [x.lower() for x in raw_text.split()]
	wordlist_no_stop = [word for word in tokenized_words if word not in stopset]
	return ' '.join(wordlist_no_stop)


def text_to_wordlist_no_punc(long_string):
	return list_to_string(tokenizer.tokenize(long_string))


def list_to_string(text_list):
	# input list of words
	# output long string
	return ' '.join(text_list)


def chunks(lst):
	text_piece_n = 100
	# for i in range(0, len(lst), text_piece_n):
	# 	yield combine_as_long_string(lst[i:i + text_piece_n])
	k, m = divmod(len(lst), text_piece_n)
	return [list_to_string(lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]) for i in range(text_piece_n)]


def map_vectorization_on_batch(list_of_text):
	return list(map(bert_text_encode, list_of_text))


def glove_lookup(word):
	try:
		word_vec = glove_model[word]
	except KeyError:
		word_vec = np.random.normal(scale=0.6, size=(glove_dim,))
	return word_vec


def doc_emb(word_list):
	vec_list = np.array(
		list(map(glove_lookup, word_list.split()))
	)
	doc_vec = np.mean(vec_list, axis=0)
	return doc_vec



def get_key_words(word_list):
	textranker.analyze(word_list, candidate_pos=['NOUN', 'PROPN', 'VERB'], window_size=4, lower=True)
	# textranker.get_keywords(10)
	key_words_list, _ = textranker.return_keywords(keyword_num)
	return key_words_list


def get_key_words_onlist(sentlist):
	all_sent_list = []
	for i in range(len(sentlist)):
		all_sent_list.append(
			list_to_string(get_key_words(sentlist[i]))
		)
		if i % 100 == 0:
			print('Processed ', i, '-th sample.')

	return all_sent_list


# to access an element in df, df.iloc[row_index][col_label]

with open(f_baseline_filename, "a") as f_baseline:
	for i in range(len(all_files)):
		f_baseline.write(all_files[i])
		f_baseline.write('\n')
		all_data = pd.read_csv(all_files[i], sep='\t')
		data_batch = all_data

		partition = int(0.9 * len(data_batch.index))

		label_batch = data_batch['label']
		text_batch = data_batch['text']

		# reconstruct each text into long list of words
		# whole dataset text_list = num of article * one big word strings
		print('get text list')
		text_list = list(map(text_to_wordlist_no_stop, text_batch))

		# remove punctuation
		text_list = list(map(text_to_wordlist_no_punc, text_list))
		text_list = list(map(text_to_wordlist_no_stop, text_list))


		##### #key word method
		# cur_time = time.time()
		# text_list = get_key_words_onlist(text_list)
		# print('get key words list takes time:', time.time() - cur_time)


		### word vector glove
		text_doc_emb_list = np.array(list(map(doc_emb, text_list)))
		X = text_doc_emb_list

		### NN try
		X = X.astype(np.float32)

		# label_batch_digits = label_to_one_hot(label_batch, n_classes)
		label_batch_digits = label_to_digits(label_batch)

		y = label_batch_digits[0:partition]
		y_true = label_batch_digits[partition:]


		class MyModule(nn.Module):
			# def __init__(self, num_units=100, nonlin=F.relu):
			def __init__(self, num_units=100, nonlin=F.relu):
				super(MyModule, self).__init__()

				self.dense0 = nn.Linear(glove_dim, num_units)
				self.nonlin = nonlin
				self.dropout = nn.Dropout(0.25)
				self.dense1 = nn.Linear(num_units, num_units)
				self.dense2 = nn.Linear(num_units, num_units)
				self.dense3 = nn.Linear(num_units, num_units)
				self.output = nn.Linear(num_units, 4)

			def forward(self, X, **kwargs):
				X = self.nonlin(self.dense0(X))
				X = self.dropout(X)
				# X = F.relu(self.dense1(X))
				X = F.sigmoid(self.dense1(X))
				X = self.dropout(X)
				X = F.sigmoid(self.dense2(X))
				X = self.dropout(X)
				X = F.sigmoid(self.dense3(X))
				X = F.softmax(self.output(X), dim=-1)
				return X


		net = NeuralNetClassifier(
			MyModule,
			max_epochs=10,
			criterion=nn.CrossEntropyLoss,
			optimizer=torch.optim.Adam,
			# Shuffle training data on each epoch
			iterator_train__shuffle=True,
			device=device,
		)

		net.fit(torch.from_numpy(X[0:partition]).float(),
				# torch.from_numpy(y.long()))
				y.long())
		y_pred = net.predict_proba(torch.from_numpy(X[partition:]).float().to(device))

		clas_report = classification_report(y_true.numpy(), np.argmax(y_pred, axis=1))
		print(clas_report)
		f_baseline.write(clas_report)
		f_baseline.write('\n')

import time
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch import tensor
from sentence_transformers import SentenceTransformer

from sklearn.metrics import f1_score

# import model
import att_model

torch.manual_seed(213)
if torch.cuda.is_available():
	print("WARNING: You have a CUDA device")

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

# bert_path = r'D:\Research\Remote\MPAA\Severity\bert_model\roberta-base-nli-mean-tokens'
bert_path = r'./bert_model/bert-base-nli-max-tokens'

bert_model = SentenceTransformer(bert_path)

# train 2911 samples
text_piece = 100  # if change, also modify in chunks()
batch_size = 50
n_classes = 4
label_dict = {'__label__0': 0, '__label__1': 1, '__label__2': 2, '__label__3': 3}

epochs = 6
log_interval = 1
lr = 0.05
clip = 0.25

'''Model Param'''
output_size = n_classes
embedding_dim = 768
hidden_dim = 768
n_layers = 1

data1 = pd.read_csv('train_frightening.csv', sep='\t')
# data2 = pd.read_csv('test_frightening.csv', sep='\t')
data3 = pd.read_csv('dev_frightening.csv', sep='\t')

train_data = data1
dev_data = data3



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


def bert_text_encode(text):
	# text: list of string sentences
	return bert_model.encode(text)


def map_vectorization_on_batch(list_of_text):
	return list(map(bert_text_encode, list_of_text))


'''Model'''
# LSTM = model.LSTMmodel(output_size=output_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, n_layers=n_layers,
# 					   drop_prob=0.5).to(device)
LSTM = att_model.AttentionModel(batch_size=batch_size, output_size=output_size, hidden_size=hidden_dim,
								embedding_dim=embedding_dim).to(device)
print(LSTM)


def repackage_hidden(h):
	"""Wraps hidden states in new Tensors, to detach them from their history."""

	if isinstance(h, torch.Tensor):
		return h.detach()
	else:
		return tuple(repackage_hidden(v) for v in h)




# evaluation
with open('att_lstm_model.pt', 'rb') as f:
	prev_model = torch.load(f)
	prev_model.eval()

	# for i in range(0, 300, batch_size):
	for i in range(0, 300, 300):
		# for i in range(0, len(train_data.index), batch_size):
		# to access an element in df, df.iloc[row_index][col_label]
		# print(i)
		data_batch = dev_data.iloc[i:i + batch_size]

		label_batch = data_batch['label']
		text_batch = data_batch['text']

		# reconstruct text into list of sentences, each sentence = long word strings, already combined
		# batch_size * arbitrary doc length
		text_batch_rec = list(map(reconstruct_text, text_batch))

		# chunk text into text_piece num of paragraphs, batch_size * piece_length (num of sent)
		text_batch_chunk = list(map(chunks, text_batch_rec))

		# training_set: batch of list of text segment embeddings, batch_size * num_of_sent * emb_dim
		dev_batch = torch.tensor(map_vectorization_on_batch(text_batch_chunk)).to(device)

		# target: batch of one-hot tensors
		target = label_to_digits(label_batch).to(device)

		output = prev_model(dev_batch)

		output_label = torch.argmax(output, dim=1)

		eval_f1 = f1_score(target.cpu(), output_label.cpu(), average='macro')

		print(eval_f1)




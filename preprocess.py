import numpy as np
import random

max_sent_length = 0
w2id = {}
curr_word_id = 1
w2id['<UNK>'] = 0
vocabulary_size = 1
PADDING_SYMBOL = 0

def get_ids(sent):

	global w2id
	global curr_word_id
	global vocabulary_size

	words = sent.strip().lower().split()

	sent_ids = []

	for w in words:
		if w not in w2id:
			w2id[w] = curr_word_id
			curr_word_id += 1
			vocabulary_size += 1

		sent_ids.append(w2id[w])
		

	return sent_ids

def get_data(training_file_path, test_file_path):

	global max_sent_length

	counter = 0 

	train_x_1 = []
	train_x_2 = []
	train_len_x_1 = []
	train_len_x_2 = []
	train_y = []

	test_x_1 = []
	test_x_2 = []
	test_len_x_1 = []
	test_len_x_2 = []
	test_y = []


	text_to_class = {	'CONTRADICTION': 0,
						'ENTAILMENT': 1,
						'NEUTRAL': 2, 
						'CONTRADICTION'.lower(): 0,
						'ENTAILMENT'.lower(): 1,
						'NEUTRAL'.lower(): 2
					}

	for line in open(training_file_path, 'r'):
		counter += 1
		#if counter == 1: continue

		parts = line.strip().split('\t')

		sent1 = parts[1]
		sent2 = parts[2]

		if parts[0] not in text_to_class:
			continue

		judgement = text_to_class[parts[0]]

		sent1_ids = get_ids(sent1)
		sent2_ids = get_ids(sent2)

		train_x_1.append(sent1_ids)
		train_x_2.append(sent2_ids)
		train_len_x_1.append(len(sent1_ids))
		train_len_x_2.append(len(sent2_ids))
		train_y.append(judgement)

	counter = 0 

	for line in open(test_file_path, 'r'):
		counter += 1
		#if counter == 1: continue

		parts = line.strip().split('\t')

		sent1 = parts[1]
		sent2 = parts[2]

		if parts[0] not in text_to_class:
			continue

		judgement = text_to_class[parts[0]]

		sent1_ids = get_ids(sent1)
		sent2_ids = get_ids(sent2)

		test_x_1.append(sent1_ids)
		test_x_2.append(sent2_ids)
		test_len_x_1.append(len(sent1_ids))
		test_len_x_2.append(len(sent2_ids))
		test_y.append(judgement)

	max_len = max([max(train_len_x_1), max(train_len_x_2), max(test_len_x_1), max(test_len_x_2)])

	for i in xrange(len(train_x_1)):
		while len(train_x_1[i]) < max_len:
			train_x_1[i].append(PADDING_SYMBOL)
		while len(train_x_2[i]) < max_len:
			train_x_2[i].append(PADDING_SYMBOL)

	for i in xrange(len(test_x_1)):
		while len(test_x_1[i]) < max_len:
			test_x_1[i].append(PADDING_SYMBOL)
		while len(test_x_2[i]) < max_len:
			test_x_2[i].append(PADDING_SYMBOL)


	print 'Maximum sentence length: %d' % max_len
	print 'Number of training samples: %d' % (len(train_x_1))
	print 'Number of test samples: %d' % (len(test_x_1))
	print 'Vocabulary size: %d' % vocabulary_size
	max_sent_length = max_len

	return np.array(train_x_1), np.array(train_x_2), np.array(train_len_x_1), np.array(train_len_x_2), np.array(train_y), np.array(test_x_1), np.array(test_x_2), np.array(test_len_x_1), np.array(test_len_x_2), np.array(test_y)


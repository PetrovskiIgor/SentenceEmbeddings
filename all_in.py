import numpy as np
import random
import tensorflow as tf

#sick_path = '/Users/igorpetrovski/Desktop/ETH/MasterThesis/skip_thoughts_data/data/sick_train/SICK_train.txt'
#sick_path = '/Users/igorpetrovski/Desktop/ETH/MasterThesis/datasets/SICK/SICK.txt'
#snli_path = '/Users/igorpetrovski/Desktop/ETH/MasterThesis/datasets/snli_1.0/snli_1.0_train.txt'
snli_train_path = '/datasets/snli/train.txt'
snli_test_path = '/datasets/snli/test.txt'
counter = 0

batch_size = 32
num_classes = 3
vocabulary_size = 0
embedding_dim = 128
max_sent_length = -1

PADDING_SYMBOL = 0

w2id = {}
curr_word_id = 0
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

	max_sent_length = max_len

	return np.array(train_x_1), np.array(train_x_2), train_len_x_1, train_len_x_2, train_y, np.array(test_x_1), np.array(test_x_2), test_len_x_1, test_len_x_2, test_y

def get_sentence_representation(embeddings, train_inputs, mask):
	lookup_op = tf.nn.embedding_lookup(embeddings, train_inputs) # TESTED
	print 'Lookup op:'
	print lookup_op.shape

	print 'Mask shape:'
	print mask.shape

	mult_op = tf.multiply(lookup_op, mask) # TESTED
	print 'mult_op:'
	print mult_op.shape

	nominator_op = tf.reduce_sum(mult_op, axis = 1) # TESTED
	denominator_op = tf.reduce_sum(mask, axis = 1) #TESTED 
	sentence_representation = tf.truediv(nominator_op, denominator_op) #TESTED


	print 'nominator_op:'
	print nominator_op.shape
	print 'denominator_op:'
	print denominator_op.shape
	print 'sentence_representation:'
	print sentence_representation.shape

	return sentence_representation

def build_model(train_inputs_1, train_inputs_2, mask_1, mask_2, train_outputs, keep_prob, layers_dim=[512]):

	embeddings = tf.Variable(tf.random_uniform(shape = [vocabulary_size, embedding_dim], minval=-1.0, maxval=1.0))

	W_1 = tf.Variable(tf.random_uniform(shape = [embedding_dim + embedding_dim + 2*embedding_dim, layers_dim[0]], minval = -1.0, maxval = 1.0))
	b_1 = tf.Variable(tf.random_uniform(shape = [layers_dim[0]], minval = -1.0, maxval = 1.0))

	W_2 = tf.Variable(tf.random_uniform(shape=[layers_dim[0], num_classes],minval=-1.0, maxval=1.0))
	b_2 = tf.Variable(tf.random_uniform(shape = [num_classes], minval=-1.0, maxval=1.0))

	first_sent = get_sentence_representation(embeddings, train_inputs_1, mask_1)
	second_sent = get_sentence_representation(embeddings, train_inputs_2, mask_2)
	
	component_1 = tf.abs(first_sent - second_sent)
	component_2 = tf.multiply(first_sent, second_sent)
	component_3 = tf.concat([first_sent, second_sent], axis = 1)
	input_representation = tf.concat([component_1, component_2, component_3], axis = 1)


	h1 = tf.nn.sigmoid(tf.add(tf.matmul(input_representation, W_1),b_1))
	h1_drop = tf.nn.dropout(h1, keep_prob)
	logits = tf.add(tf.matmul(h1_drop, W_2),b_2)

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = train_outputs, logits = logits))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.5)
	train_step = optimizer.minimize(loss)

	return train_step, loss, logits


def train_test_split(X_1, X_2, y_transformed, mask_1, mask_2, N, train_test_ratio = 0.9):
	indices = np.arange(N)
	random.shuffle(indices)

	X_1 = X_1[indices]
	X_2 = X_2[indices]
	y_transformed = y_transformed[indices]
	mask_1 = mask_1[indices]
	mask_2 = mask_2[indices]
	
	train_size = int(train_test_ratio * N)

	print 'N: %d' % N
	print 'Train size: %d' % train_size

	return	X_1[:train_size],X_2[:train_size], y_transformed[:train_size], mask_1[:train_size],mask_2[:train_size],X_1[train_size:],X_2[train_size:], y_transformed[train_size:], mask_1[train_size:],mask_2[train_size:]



def train():

	X_1, X_2, len_x_1, len_x_2, y, X_1_test, X_2_test, len_x_1_test, len_x_2_test, y_test = get_data(training_file_path = snli_train_path, test_file_path = snli_test_path)

	N_train = len(X_1)

	assert N_train == len(X_2) and N_train == len(len_x_1) and N_train == len(len_x_2) and N_train == len(y)

	N_test = len(X_1_test)

	assert N_test == len(X_2_test) and N_test == len(len_x_1_test) and N_test == len(len_x_2_test) and N_test == len(y_test)


	
	y_transformed = np.zeros((len(y), num_classes))
	y_transformed[np.arange(len(y)), y] = 1

	y_transformed_test = np.zeros((len(y_test), num_classes))
	y_transformed_test[np.arange(len(y_test)), y_test] = 1

	tensor_len_x_1 = tf.placeholder(tf.int32, shape = [None])
	tensor_len_x_2 = tf.placeholder(tf.int32, shape = [None])

	mask_1 = tf.cast(tf.sequence_mask(tensor_len_x_1, maxlen = max_sent_length), tf.float32)
	mask_2 = tf.cast(tf.sequence_mask(tensor_len_x_2, maxlen = max_sent_length), tf.float32)

	train_inputs_1 = tf.placeholder(tf.int32, shape = [None, max_sent_length])
	train_inputs_2 = tf.placeholder(tf.int32, shape = [None, max_sent_length])
	train_mask_inputs_1 = tf.placeholder(tf.float32, shape = [None, max_sent_length])
	train_mask_inputs_2 = tf.placeholder(tf.float32, shape = [None, max_sent_length])
	train_mask_inputs_reshaped_1 = tf.reshape(train_mask_inputs_1, shape = [-1, max_sent_length, 1])
	train_mask_inputs_reshaped_2 = tf.reshape(train_mask_inputs_2, shape = [-1, max_sent_length, 1])
	train_outputs = tf.placeholder(tf.float32, shape = [None, num_classes])
	keep_prob = tf.placeholder(tf.float32, name = 'Dropout')

	
	
	train_step, loss, predicted = build_model(train_inputs_1, train_inputs_2, train_mask_inputs_reshaped_1, train_mask_inputs_reshaped_2, train_outputs, keep_prob)
	
	y_true = tf.placeholder(tf.float32, shape = [None, num_classes])
	accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predicted, axis = 1), tf.argmax(y_true, axis = 1)), tf.float32))
	print 'Vocabulary size: %d' % vocabulary_size

	with tf.Session() as sess:

		sess.run(tf.global_variables_initializer())

		mask1_instance = sess.run(mask_1, feed_dict = {
			tensor_len_x_1: len_x_1
		})
		mask2_instance = sess.run(mask_2, feed_dict = {
			tensor_len_x_2: len_x_2
		})
		
		mask_1_instance_test, mask_2_instance_test = sess.run([mask_1,mask_2], feed_dict = {
			tensor_len_x_1: len_x_1_test,
			tensor_len_x_2: len_x_2_test
		})
		
		#X_1, X_2, y_transformed, mask_1_instance, mask_2_instance, X_1_test, X_2_test, y_transformed_test, mask_1_test, mask_2_test = train_test_split(np.array(X_1), np.array(X_2), np.array(y_transformed), np.array(mask1_instance), np.array(mask2_instance), N, 0.9)


		print 'X_1_test shape: ', X_1_test.shape
		print 'X_2_test shape: ', X_2_test.shape
		print 'y_transformed_test shape: ', y_transformed_test.shape
		print 'mask_1_instance_test shape:', mask_1_instance_test.shape
		print 'mask_2_instance_test shape:', mask_2_instance_test.shape
		avg_loss = 0.0

		i = 0
		for epoch in xrange(100):
			ind = 0
			print 'Epoch: %d' % epoch
			while ind < len(X_1):
				i += 1
				max_ind = min(ind + batch_size, len(X_1))
				batch_x_1 = X_1[ind:max_ind]
				batch_x_2 = X_2[ind:max_ind]
				batch_x_len_1 = mask1_instance[ind:max_ind]
				batch_x_len_2 = mask2_instance[ind:max_ind]
				batch_y = y_transformed[ind:max_ind]

				if max_ind == len(X_1):
					batch_x_1 = np.vstack((batch_x_1,X_1[:batch_size - len(batch_x_1)]))

					batch_x_2 = np.vstack((batch_x_2,X_2[:batch_size - len(batch_x_2)]))
					batch_x_len_1 = np.vstack((batch_x_len_1,mask1_instance[:batch_size - len(batch_x_len_1)]))
					batch_x_len_2 = np.vstack((batch_x_len_2,mask2_instance[:batch_size - len(batch_x_len_2)]))
					batch_y = np.vstack((batch_y,y_transformed[:batch_size - len(batch_y)]))

				_, curr_loss = sess.run([train_step, loss], feed_dict = {
					train_outputs: batch_y,
					train_inputs_1: batch_x_1,
					train_inputs_2: batch_x_2, 
					train_mask_inputs_1: batch_x_len_1,
					train_mask_inputs_2: batch_x_len_2, 
					keep_prob: 0.5
					
				})
				avg_loss += curr_loss

				if ind % 10000 == 0 and i > 0:
					accuracy_instance = sess.run(accuracy, feed_dict = {
						train_inputs_1: X_1_test,
						train_inputs_2: X_2_test, 
						train_mask_inputs_1: mask_1_instance_test,
						train_mask_inputs_2: mask_2_instance_test,
						y_true: y_transformed_test, 
						keep_prob: 1.0
					})

					print '\tIteration: %i Average loss: %.2f Accuracy: %.2f' % (ind, avg_loss / i, accuracy_instance)

				ind += batch_size
	

train()

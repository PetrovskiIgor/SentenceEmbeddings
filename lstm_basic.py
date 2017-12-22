import numpy
import tensorflow as tf
import math
import random
from preprocess import get_data
from preprocess import get_ids
from preprocess import max_sent_length
from preprocess import vocabulary_size

batch_size = 32
max_len = 30
embedding_dim = 128
lstm_num_units = 128
num_classes = 3
hidden_state_dim = 256
snli_train_path = '/datasets/snli/train.txt'
snli_test_path = '/datasets/snli/test.txt'


def encode_sentence(inputs,sequence_length, num_units=128):
	lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units)
	outputs, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = inputs, dtype = tf.float32, sequence_length = sequence_length)
	return state

def build_model(inputs_1, inputs_2, seq_len_1, seq_len_2, labels, keep_prob = 0.5, fc_layers_dim = [512]):
	embeddings = tf.Variable(tf.random_uniform(shape = [vocabulary_size, embedding_dim], minval=-1.0, maxval=1.0))
	first_sent_embeddings = tf.nn.embedding_lookup(embeddings, inputs_1)
	second_sent_embeddings = tf.nn.embedding_lookup(embeddings, inputs_2)

	with tf.variable_scope('lstm1'):
		_,first_sent = encode_sentence(inputs = first_sent_embeddings, sequence_length = seq_len_1, num_units = hidden_state_dim)

	with tf.variable_scope('lstm2'):
		_,second_sent = encode_sentence(inputs = second_sent_embeddings, sequence_length = seq_len_2, num_units = hidden_state_dim)

	component_1 = tf.abs(first_sent - second_sent)
	component_2 = tf.multiply(first_sent, second_sent)
	component_3 = tf.concat([first_sent, second_sent], axis = 1)
	input_representation = tf.concat([component_1, component_2, component_3], axis = 1)

	W_1 = tf.Variable(tf.random_uniform(shape = [hidden_state_dim + hidden_state_dim + 2*hidden_state_dim, fc_layers_dim[0]], minval = -1.0, maxval = 1.0))
	b_1 = tf.Variable(tf.random_uniform(shape = [fc_layers_dim[0]], minval = -1.0, maxval = 1.0))

	W_2 = tf.Variable(tf.random_uniform(shape=[fc_layers_dim[0], num_classes],minval=-1.0, maxval=1.0))
	b_2 = tf.Variable(tf.random_uniform(shape = [num_classes], minval=-1.0, maxval=1.0))


	h1 = tf.nn.sigmoid(tf.add(tf.matmul(input_representation, W_1),b_1))
	h1_drop = tf.nn.dropout(h1, keep_prob)
	logits = tf.add(tf.matmul(h1_drop, W_2),b_2)

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.5)
	train_step = optimizer.minimize(loss)

	return train_step, loss, logits

def train():

	max_sent_length,X_1, X_2, len_x_1, len_x_2, y, X_1_test, X_2_test, len_x_1_test, len_x_2_test, y_test = get_data(training_file_path = snli_train_path, test_file_path = snli_test_path)

	N_train = len(X_1)
	N_test = len(X_1_test)

	assert N_train == len(X_2) and N_train == len(len_x_1) and N_train == len(len_x_2) and N_train == len(y)
	assert N_test == len(X_2_test) and N_test == len(len_x_1_test) and N_test == len(len_x_2_test) and N_test == len(y_test)

	y_transformed = np.zeros((len(y), num_classes))
	y_transformed[np.arange(len(y)), y] = 1

	y_transformed_test = np.zeros((len(y_test), num_classes))
	y_transformed_test[np.arange(len(y_test)), y_test] = 1

	tensor_len_x_1 = tf.placeholder(tf.int32, shape = [None])
	tensor_len_x_2 = tf.placeholder(tf.int32, shape = [None])

	train_inputs_1 = tf.placeholder(tf.int32, shape = [None, max_sent_length])
	train_inputs_2 = tf.placeholder(tf.int32, shape = [None, max_sent_length])
	train_outputs = tf.placeholder(tf.float32, shape = [None, num_classes])
	"""
	keep_prob = tf.placeholder(tf.float32, name = 'Dropout')

	train_step, loss, predicted = build_model(train_inputs_1, train_inputs_2, train_mask_inputs_reshaped_1, train_mask_inputs_reshaped_2, train_outputs, keep_prob)
	
	y_true = tf.placeholder(tf.float32, shape = [None, num_classes])
	accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predicted, axis = 1), tf.argmax(y_true, axis = 1)), tf.float32))

	print 'Vocabulary size: %d' % vocabulary_size

	with tf.Session() as sess:

		sess.run(tf.global_variables_initializer())
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
						train_mask_inputs_1: mask_1_test,
						train_mask_inputs_2: mask_2_test,
						y_true: y_transformed_test, 
						keep_prob: 1.0
					})

					print '\tIteration: %i Average loss: %.2f Accuracy: %.2f' % (ind, avg_loss / i, accuracy_instance)

				ind += batch_size
	"""

train()
"""
inputs = tf.placeholder(dtype = tf.int32, shape = [batch_size, max_len])
sequence_length = tf.placeholder(dtype=tf.int32, shape = [batch_size])
labels = tf.placeholder(dtype = tf.int32, shape = [batch_size, num_classes])
train_step, loss, logits = build_model(inputs_1 = inputs, inputs_2 = inputs, seq_len_1 = sequence_length, seq_len_2 = sequence_length, labels = labels)
"""
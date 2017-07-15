from __future__ import division
from __future__ import print_function
import argparse
import time
import random

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from ptb_reader import ptb_raw_data
from ptb_reg import PTBRegModel

# Adapted from https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py


class SmallConfig(object):
	max_grad_norm = 5
	num_layers = 2
	num_steps = 20
	hidden_size = 200
	#max_epoch = 4
	keep_prob = 1.0
	batch_size = 20
	vocab_size = 10000


class MediumConfig(object):
	max_grad_norm = 5
	num_layers = 2
	num_steps = 35
	hidden_size = 650
	#max_epoch = 6
	keep_prob = 0.5
	batch_size = 20
	vocab_size = 10000


class LargeConfig(object):
	max_grad_norm = 10
	num_layers = 2
	num_steps = 35
	hidden_size = 1500
	#max_epoch = 14
	keep_prob = 0.35
	batch_size = 20
	vocab_size = 10000
	

class PTBModel(object):
	def __init__(self,config):
		self.c = c = config
		
		self.x = tf.placeholder(tf.int32, [c.batch_size, c.num_steps], 'x')
		self.y = tf.placeholder(tf.int32, [c.batch_size, c.num_steps], 'y')
		
		if c.reg_type == 'static':
			self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
		
		with tf.device("/cpu:0"):
			embedding = tf.get_variable("embedding", [c.vocab_size, c.hidden_size], dtype=tf.float32)
			h = tf.nn.embedding_lookup(embedding, self.x)
			
		if c.reg_type == 'static':
			h = tf.nn.dropout(h, self.keep_prob)
			
		cells = [tf.contrib.rnn.LSTMCell(num_units = c.hidden_size) for i in range(c.num_layers)]
		
		if c.reg_type == 'static':
			cells = [tf.contrib.rnn.DropoutWrapper(i, output_keep_prob=self.keep_prob) for i in cells]
		
		cells = tf.contrib.rnn.MultiRNNCell(cells)

		h = tf.unstack(h, num=c.num_steps, axis=1)
		output, state = tf.contrib.rnn.static_rnn(cells, h, dtype=tf.float32)
		
		h = tf.reshape(tf.stack(axis=1, values=output), [-1, c.hidden_size])
		
		logits = tf.contrib.layers.fully_connected(h, c.vocab_size, activation_fn=tf.identity)
		
		# Reshape logits to be 3-D tensor for sequence loss
		logits = tf.reshape(logits, [c.batch_size, c.num_steps, c.vocab_size])
		
		loss = tf.contrib.seq2seq.sequence_loss(
			logits,
			self.y,
			tf.ones([c.batch_size, c.num_steps], dtype=tf.float32),
			average_across_timesteps=False,
			average_across_batch=True
		)
		self.loss = tf.reduce_sum(loss)
		
		optimizer = tf.train.AdamOptimizer()
		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), c.max_grad_norm)	
		self.train_step = optimizer.apply_gradients(zip(grads, tvars),
			global_step=tf.contrib.framework.get_or_create_global_step())

		
	def fit(self, train_data, val_data, sess):
		c = self.c

		results = []
		for epoch in range(100):
			print("\nEpoch %d" % (epoch+1))
			start = time.time()
			offset = random.randrange(c.num_steps)
			train_indices = list(range(offset, len(train_data)-c.num_steps, c.num_steps))
			random.shuffle(train_indices)
			train_indices = [train_indices[i:i + c.batch_size] for i in range(0,len(train_indices),c.batch_size)] # Batch the data
			train_indices = train_indices[:-1]

			total_loss = 0.0
			total_iters = 0.0
			
			for it in train_indices:
				batch_train_x = [train_data[i:i+c.num_steps+1] for i in it]
				batch_train_y = [i[1:] for i in batch_train_x]
				batch_train_x = [x[:-1] for x in batch_train_x]
				
				feed_dict = {self.x: batch_train_x, 
							self.y: batch_train_y}
					
				if c.reg_type == 'static':
					feed_dict[self.keep_prob] = c.keep_prob

				_,loss = sess.run([self.train_step,self.loss], feed_dict)
				
				total_loss += loss
				total_iters += c.num_steps

			train_perplexity = np.exp(total_loss/total_iters)
			print("Train perplexity: %.3f" % train_perplexity)
			
			val_perplexity = self.eval_perplexity(val_data,c,sess)
			print("Val perplexity: %.3f" % val_perplexity)
			
			print("Time taken: %.3f" % (time.time() - start))

			
	def eval_perplexity(self,data,c,sess):
		indices = list(range(0, len(data)-c.num_steps, c.num_steps))
		indices = [indices[i:i + c.batch_size] for i in range(0,len(indices),c.batch_size)] # Batch the data
		indices = indices[:-1]
		
		total_loss = 0.0
		total_iters = 0.0
		
		for it in indices:
				batch_x = [data[i:i+c.num_steps+1] for i in it]
				batch_y = [i[1:] for i in batch_x]
				batch_x = [x[:-1] for x in batch_x]
				
				feed_dict = {self.x: batch_x, 
							self.y: batch_y}
							
				if c.reg_type == 'static':
					feed_dict[self.keep_prob] = 1.0

				loss = sess.run(self.loss, feed_dict)
				
				total_loss += loss
				total_iters += c.num_steps
				
		return np.exp(total_loss/total_iters)
		

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--reg', type=str, help='none, static or adaptive', required=True)
	args = parser.parse_args()

	X_train, X_valid, X_test, vocab = ptb_raw_data()
	
	print("\nData loaded")
	print("Training set: %d words" % len(X_train))
	print("Validation set: %d words" % len(X_valid))
	print("Test set: %d words" % len(X_test))
	print("Vocab size: %d words\n" % len(vocab))
		
	c = SmallConfig()
	if args.reg == 'none':
		c.keep_prob = 1.0
		
	if args.reg == 'adaptive':
		m = PTBRegModel(c)
	else:
		c.reg_type = args.reg
		m = PTBModel(c)
	
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	
	m.fit(X_train,X_valid,sess)	
	m.eval_perplexity(X_test)
		
if __name__ == "__main__":
	main()

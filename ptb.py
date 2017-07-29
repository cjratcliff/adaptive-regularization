from __future__ import division
from __future__ import print_function
import argparse
import time
import random
import copy

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from ptb_reader import ptb_raw_data
from custom_lstm import CustomLSTMCell, CustomMultiRNNCell

# Adapted from https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py


class SmallConfig(object):
	max_grad_norm = 5
	num_layers = 2
	num_steps = 20
	hidden_size = 200
	#max_epochs = 13
	keep_prob = 1.0
	batch_size = 20
	decay_lr_at = 4
	lr_decay = 0.5
	vocab_size = 10000
	wd_lr = 0.001
	wd_clipping = 0.1

class MediumConfig(object):
	max_grad_norm = 5
	num_layers = 2
	num_steps = 35
	hidden_size = 650
	#max_epochs = 39
	keep_prob = 0.5
	batch_size = 20
	decay_lr_at = 6
	lr_decay = 0.8	
	vocab_size = 10000
	wd_lr = 0.001
	wd_clipping = 0.01

class LargeConfig(object):
	max_grad_norm = 10
	num_layers = 2
	num_steps = 35
	hidden_size = 1500
	#max_epochs = 55
	keep_prob = 0.35
	batch_size = 20
	decay_lr_at = 14
	lr_decay = 1 / 1.15
	vocab_size = 10000
	wd_lr = 0.001 # Needs to be tuned
	wd_clipping = 0.002 # Needs to be tuned
	

class PTBModel(object):
	def __init__(self,config):
		self.c = c = config
		
		self.x = tf.placeholder(tf.int32, [None, None], 'x')
		self.y = tf.placeholder(tf.int32, [None, None], 'y')
		
		if c.reg_type == 'adaptive':
			self.val_x = tf.placeholder(tf.int32, [None, None], 'val_x')
			self.val_y = tf.placeholder(tf.int32, [None, None], 'val_y')
			
		self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
		
		if c.reg_type == 'static':
			self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
		else:
			self.keep_prob = None
			
		cells = [tf.contrib.rnn.LSTMCell(num_units = c.hidden_size) for i in range(c.num_layers)]				
		if c.reg_type == 'static':
			cells = [tf.contrib.rnn.DropoutWrapper(i, output_keep_prob=self.keep_prob) for i in cells]
		cells = tf.contrib.rnn.MultiRNNCell(cells)
		
		self.initial_state = cells.zero_state(self.batch_size, tf.float32)

		logits,self.final_state = self.forward_prop(self.x, cells, self.initial_state, keep_prob=self.keep_prob)
		self.loss = self.loss_fn(logits,self.y)
		
		main_params = tf.trainable_variables()
		
		if c.reg_type == 'adaptive':
			self.l2_weight_decay_coef = tf.Variable(tf.constant(0.0), name="l2_weight_decay")
			l2_weight_decay = sum([tf.reduce_sum(tf.square(i)) for i in main_params])
			self.loss_reg = self.loss + tf.maximum(0.0,self.l2_weight_decay_coef)*l2_weight_decay
		
		self.lr = tf.Variable(tf.constant(1.0),trainable=False)
		optimizer = tf.train.GradientDescentOptimizer(self.lr)

		if c.reg_type == 'adaptive':
			grads = tf.gradients(self.loss_reg, main_params)	
		else:
			grads = tf.gradients(self.loss, main_params)
			
		grads,_ = tf.clip_by_global_norm(grads, c.max_grad_norm)
		gv = list(zip(grads,main_params))
		self.train_step = optimizer.apply_gradients(gv)

		if c.reg_type == 'adaptive':
			# Compute gradients on the training sample with weight decay
			train_update = [tf.multiply(-self.lr,g) for (g,v) in gv]
			new_params = [v+u for (v,u) in zip(main_params,train_update)] 

			cells = [CustomLSTMCell(num_units = c.hidden_size) for i in range(c.num_layers)]
			cells = CustomMultiRNNCell(cells)
			self.val_initial_state = cells.zero_state(self.batch_size, tf.float32)
			
			val_logits,self.val_final_state = self.forward_prop(self.val_x, cells, self.val_initial_state, new_params)		
			self.val_loss = self.loss_fn(val_logits,self.val_y)

			term1 = tf.gradients([self.val_loss],train_update)
			term2 = [-2*self.lr*v for v in main_params]

			reg_grad = 0.0
			for (v1,v2) in zip(term1,term2):
				reg_grad += tf.reduce_sum(v1*v2)
			reg_grad = tf.clip_by_value(reg_grad, -c.wd_clipping, c.wd_clipping)
			gv = [(reg_grad,self.l2_weight_decay_coef)]
			self.reg_train_step = tf.train.GradientDescentOptimizer(c.wd_lr).apply_gradients(gv)
					
			
	def loss_fn(self,logits,y):
		loss = tf.contrib.seq2seq.sequence_loss(
				logits,
				y,
				tf.ones([self.c.batch_size, self.c.num_steps], dtype=tf.float32),
				average_across_timesteps=False,
				average_across_batch=True
		)
		return tf.reduce_sum(loss)
		
		
	def forward_prop(self, x, cells, state, params=None, keep_prob=None,):
		c = self.c
		
		with tf.device("/cpu:0"):
			if params is None:
				embedding = tf.get_variable("embedding", [c.vocab_size, c.hidden_size], dtype=tf.float32)
			else:
				embedding = params[0]
			h = tf.nn.embedding_lookup(embedding, x)
			
		if c.reg_type == 'static':
			h = tf.nn.dropout(h, keep_prob)
			
		h = tf.unstack(h, num=c.num_steps, axis=1)
		
		outputs = []
		with tf.variable_scope("RNN"):
			for time_step in range(c.num_steps):
				if time_step > 0:
					tf.get_variable_scope().reuse_variables()
				if params is None:
					(cell_output, state) = cells(h[time_step], state)
				else:
					(cell_output, state) = cells(h[time_step], state, [params[1:3],params[3:5]])
				outputs.append(cell_output)

		h = tf.reshape(tf.stack(axis=1, values=outputs), [-1, c.hidden_size])
		
		if params is None:
			logits = tf.contrib.layers.fully_connected(h, c.vocab_size, activation_fn=tf.identity)
		else:
			logits = tf.matmul(h,params[5]) + params[6]

		# Reshape logits to be 3-D tensor for sequence loss
		return tf.reshape(logits, [c.batch_size, c.num_steps, c.vocab_size]), state

		
	def fit(self, train_data, val_data, sess):
		c = self.c

		results = []
		for epoch in range(100):
			print("\nEpoch %d" % (epoch+1))
			start = time.time()
			
			# Decay the learning rate
			if epoch >= c.decay_lr_at:
				sess.run(tf.assign(self.lr,c.lr_decay*self.lr))
				print("Learning rate set to: %f" % sess.run(self.lr))
			
			if c.reg_type == 'adaptive':
				train_perplexity = self.run_epoch(train_data, True, False, sess, val_data=val_data)
			else:
				train_perplexity = self.run_epoch(train_data, True, False, sess)
			print("Train perplexity: %.3f" % train_perplexity)
			
			val_perplexity = self.run_epoch(val_data, False, False, sess)
			print("Val perplexity: %.3f" % val_perplexity)
			
			print("Time taken: %.3f" % (time.time() - start))
			
			results.append([train_perplexity,val_perplexity])
			np.savetxt('results.csv', np.array(results), fmt='%5.5f', delimiter=',')


	def reshape_data(self,data):
		c = self.c
		num_batches = len(data) // c.batch_size
		data = data[0 : c.batch_size * num_batches]
		data = np.reshape(data,[c.batch_size, num_batches])
		return data
		
			
	def run_epoch(self, data, is_training, full_eval, sess, val_data=None):
		assert not(is_training and full_eval)
		c = copy.deepcopy(self.c)
		
		if full_eval: # Very slow so only used for the test set
			c.batch_size = 1
			c.num_steps = 1		

		data = self.reshape_data(data)
		if c.reg_type == 'adaptive' and is_training:
			val_data = self.reshape_data(val_data)
		
		total_loss = 0.0
		total_iters = 0.0
		
		state = sess.run(self.initial_state, feed_dict={self.batch_size: c.batch_size})
		if self.c.reg_type == 'adaptive' and is_training:
			val_state = sess.run(self.val_initial_state, feed_dict={self.batch_size: c.batch_size})
				
		for idx in range(0,data.shape[1],c.num_steps):
			batch_x = data[:, idx:idx+c.num_steps]
			batch_y = data[:, idx+1:idx+c.num_steps+1]
			
			if batch_x.shape != (c.batch_size,c.num_steps) or \
				batch_y.shape != (c.batch_size,c.num_steps):
				#print(batch_x.shape,batch_y.shape)
				continue
							
			feed_dict = {self.x: batch_x, 
						self.y: batch_y,
						self.batch_size: c.batch_size}
						
			if c.reg_type == 'adaptive' and is_training:
				val_idx = random.choice(range(0,val_data.shape[1],c.num_steps))
				batch_val_x = val_data[:, val_idx:val_idx+c.num_steps]
				batch_val_y = val_data[:, val_idx+1:val_idx+c.num_steps+1]
				
				if batch_val_x.shape != (c.batch_size,c.num_steps) or \
					batch_val_y.shape != (c.batch_size,c.num_steps):
					continue						

				feed_dict[self.val_x] = batch_val_x
				feed_dict[self.val_y] = batch_val_y
						
			for i, (c_state,h_state) in enumerate(self.initial_state):
				feed_dict[c_state] = state[i].c
				feed_dict[h_state] = state[i].h
				
			if self.c.reg_type == 'adaptive' and is_training:
				for i, (c_state,h_state) in enumerate(self.val_initial_state):
					feed_dict[c_state] = val_state[i].c
					feed_dict[h_state] = val_state[i].h				
						
			if c.reg_type == 'static':
				if is_training:
					feed_dict[self.keep_prob] = c.keep_prob
				else:
					feed_dict[self.keep_prob] = 1.0

			if is_training:
				if c.reg_type == 'adaptive':
					_,_,loss,state,val_state,wd = sess.run([self.train_step, self.reg_train_step, self.loss, self.final_state, self.val_final_state, self.l2_weight_decay_coef], feed_dict)
				else: 
					_,loss,state = sess.run([self.train_step, self.loss, self.final_state], feed_dict)			
			else:
				loss,state = sess.run([self.loss,self.final_state], feed_dict)
			
			total_loss += loss
			total_iters += c.num_steps

		return np.exp(total_loss/total_iters)
		

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--reg', type=str, help='none, static or adaptive', required=True)
	parser.add_argument('--size', type=str, help='small, medium or large', required=True)
	args = parser.parse_args()

	X_train, X_valid, X_test, vocab = ptb_raw_data()
	
	print("\nData loaded")
	print("Training set: %d words" % len(X_train))
	print("Validation set: %d words" % len(X_valid))
	print("Test set: %d words" % len(X_test))
	print("Vocab size: %d words\n" % len(vocab))
		
	if args.size == 'small':
		c = SmallConfig()
	elif args.size == 'medium':
		c = MediumConfig()	
	elif args.size == 'large':
		c = LargeConfig()	
	else:
		raise ValueError("Invalid value for size argument")
	
	assert args.reg in ['none','static','adaptive']
	c.reg_type = args.reg
	m = PTBModel(c)
	
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	
	m.fit(X_train,X_valid,sess)	
	m.run_epoch(X_test, False, True, sess)
		
if __name__ == "__main__":
	main()



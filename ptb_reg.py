from __future__ import division
from __future__ import print_function
import random
import time

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from custom_lstm import CustomLSTMCell, CustomMultiRNNCell


def forward_prop(x,c,params=None):

	with tf.device("/cpu:0"):
		if params is None:
			embedding = tf.get_variable("embedding", [c.vocab_size, c.hidden_size], dtype=tf.float32)
		else:
			embedding = params[0]
		h = tf.nn.embedding_lookup(embedding, x)
		
	if params is None:
		cells = [tf.contrib.rnn.LSTMCell(num_units = c.hidden_size) for i in range(c.num_layers)]
		cells = tf.contrib.rnn.MultiRNNCell(cells)
	else:
		cells = [CustomLSTMCell(num_units = c.hidden_size) for i in range(c.num_layers)]
		cells = CustomMultiRNNCell(cells)

	h = tf.unstack(h, num=c.num_steps, axis=1)
	
	outputs = []
	state = cells.zero_state(c.batch_size, tf.float32)
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
	return tf.reshape(logits, [c.batch_size, c.num_steps, c.vocab_size])


class PTBRegModel(object):
	def __init__(self,config):
		self.c = c = config
		
		self.train_x = tf.placeholder(tf.int32, [c.batch_size, c.num_steps], 'train_x')
		self.train_y = tf.placeholder(tf.int32, [c.batch_size, c.num_steps], 'train_y')
		self.val_x = tf.placeholder(tf.int32, [c.batch_size, c.num_steps], 'val_x')
		self.val_y = tf.placeholder(tf.int32, [c.batch_size, c.num_steps], 'val_y')
	
		with tf.variable_scope('forward') as scope:
			logits = forward_prop(self.train_x,c)
		
		train_loss = tf.contrib.seq2seq.sequence_loss(
			logits,
			self.train_y,
			tf.ones([c.batch_size, c.num_steps], dtype=tf.float32),
			average_across_timesteps=False,
			average_across_batch=True
		)
		self.train_loss = tf.reduce_sum(train_loss)
		
		weight_vars = tf.trainable_variables()
		self.weight_decay = tf.Variable(tf.constant(0.0),name="weight_decay")		
		l2_weight_decay = sum([tf.reduce_sum(tf.square(i)) for i in weight_vars])
		
		self.train_loss_reg += tf.maximum(0.0,self.l2_weight_decay)*l2_weight_decay
		
		lr = 0.01
		optimizer = tf.train.AdamOptimizer(learning_rate=lr)
		gv = optimizer.compute_gradients(self.train_loss_reg, var_list=weight_vars)
		
		grads, vars = tf.clip_by_global_norm(tf.gradients(self.train_loss_reg, weight_vars), c.max_grad_norm)	
		self.train_step = optimizer.apply_gradients(zip(grads, weight_vars),
			global_step=tf.contrib.framework.get_or_create_global_step())				
		
		# Compute gradients on the training sample with weight decay
		train_update = [tf.multiply(-lr,g) for (g,v) in gv]
		new_params = [v+u for (v,u) in zip(weight_vars,train_update)]
		
		with tf.variable_scope('forward') as scope:
			scope.reuse_variables()
			val_logits = forward_prop(self.val_x, c, new_params)
		
		val_loss = tf.contrib.seq2seq.sequence_loss(
			val_logits,
			self.val_y,
			tf.ones([c.batch_size, c.num_steps], dtype=tf.float32),
			average_across_timesteps=False,
			average_across_batch=True
		)
		val_loss = tf.reduce_sum(val_loss)

		term1 = tf.gradients([val_loss],train_update) # d_L/d_delta_theta
		term2 = [-2*lr*v for v in weight_vars] # d_delta_theta/d_C2
		
		# Apply the chain rule and sum over the gradients
		reg_grad = 0.0
		for (v1,v2) in zip(term1,term2):
			reg_grad += tf.reduce_sum(v1*v2)
			
		gv = [(reg_grad_l2,self.weight_decay)]
			
		self.reg_train_step = tf.train.GradientDescentOptimizer(1e-4).apply_gradients(gv)
	
	
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

			val_indices = list(range(0, len(val_data)-c.num_steps, c.num_steps))
			val_indices = [val_indices[i:i + c.batch_size] for i in range(0,len(val_indices),c.batch_size)] # Batch the data
			val_indices = val_indices[:-1]

			total_train_loss = 0.0
			total_iters = 0.0
			
			for it in train_indices:
				batch_train_x = [train_data[i:i+c.num_steps+1] for i in it]
				batch_train_y = [i[1:] for i in batch_train_x]
				batch_train_x = [x[:-1] for x in batch_train_x]
				
				iv = random.choice(val_indices)
				batch_val_x = [val_data[i:i+c.num_steps+1] for i in iv]
				batch_val_y = [i[1:] for i in batch_val_x]
				batch_val_x = [x[:-1] for x in batch_val_x]
				
				feed_dict = {self.train_x: batch_train_x, 
							self.train_y: batch_train_y,
							self.val_x: batch_val_x,
							self.val_y: batch_val_y}

				_,_,loss = sess.run([self.train_step,self.reg_train_step,self.train_loss], feed_dict)
				
				total_train_loss += loss
				total_iters += c.num_steps

			train_perplexity = np.exp(total_train_loss/total_iters)
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
				
				feed_dict = {self.train_x: batch_x, 
							self.train_y: batch_y}

				loss = sess.run(self.train_loss, feed_dict)
				
				total_loss += loss
				total_iters += c.num_steps
				
		return np.exp(total_loss/total_iters)

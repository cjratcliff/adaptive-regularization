# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#		 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import hashlib
import numbers

import tensorflow as tf

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest


_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


def _like_rnncell(cell):
	"""Checks that a given object is an RNNCell by using duck typing."""
	conditions = [hasattr(cell, "output_size"), hasattr(cell, "state_size"),
								hasattr(cell, "zero_state"), callable(cell)]
	return all(conditions)


def _concat(prefix, suffix, static=False):
	"""Concat that enables int, Tensor, or TensorShape values.

	This function takes a size specification, which can be an integer, a
	TensorShape, or a Tensor, and converts it into a concatenated Tensor
	(if static = False) or a list of integers (if static = True).

	Args:
		prefix: The prefix; usually the batch size (and/or time step size).
			(TensorShape, int, or Tensor.)
		suffix: TensorShape, int, or Tensor.
		static: If `True`, return a python list with possibly unknown dimensions.
			Otherwise return a `Tensor`.

	Returns:
		shape: the concatenation of prefix and suffix.

	Raises:
		ValueError: if `suffix` is not a scalar or vector (or TensorShape).
		ValueError: if prefix or suffix was `None` and asked for dynamic
			Tensors out.
	"""
	if isinstance(prefix, ops.Tensor):
		p = prefix
		p_static = tensor_util.constant_value(prefix)
		if p.shape.ndims == 0:
			p = tf.expand_dims(p, 0)
		elif p.shape.ndims != 1:
			raise ValueError("prefix tensor must be either a scalar or vector, "
											 "but saw tensor: %s" % p)
	else:
		p = tensor_shape.as_shape(prefix)
		p_static = p.as_list() if p.ndims is not None else None
		p = (constant_op.constant(p.as_list(), dtype=dtypes.int32)
				 if p.is_fully_defined() else None)
	if isinstance(suffix, ops.Tensor):
		s = suffix
		s_static = tensor_util.constant_value(suffix)
		if s.shape.ndims == 0:
			s = tf.expand_dims(s, 0)
		elif s.shape.ndims != 1:
			raise ValueError("suffix tensor must be either a scalar or vector, "
											 "but saw tensor: %s" % s)
	else:
		s = tensor_shape.as_shape(suffix)
		s_static = s.as_list() if s.ndims is not None else None
		s = (constant_op.constant(s.as_list(), dtype=dtypes.int32)
				 if s.is_fully_defined() else None)

	if static:
		shape = tensor_shape.as_shape(p_static).concatenate(s_static)
		shape = shape.as_list() if shape.ndims is not None else None
	else:
		if p is None or s is None:
			raise ValueError("Provided a prefix or suffix of None: %s and %s"
											 % (prefix, suffix))
		shape = tf.concat((p, s), 0)
	return shape


def _zero_state_tensors(state_size, batch_size, dtype):
	"""Create tensors of zeros based on state_size, batch_size, and dtype."""
	def get_state_shape(s):
		"""Combine s with batch_size to get a proper tensor shape."""
		c = _concat(batch_size, s)
		c_static = _concat(batch_size, s, static=True)
		size = tf.zeros(c, dtype=dtype)
		size.set_shape(c_static)
		return size
	return nest.map_structure(get_state_shape, state_size)


class RNNCell(base_layer.Layer):

	#def __call__(self, inputs, state, scope=None):
	#	if scope is not None:
			###with tf.variable_scope(scope, custom_getter=self._rnn_get_variable) as scope:
	#		return super(RNNCell, self).__call__(inputs, state)#, scope=scope)
	#	else:
			###with tf.variable_scope(tf.get_variable_scope(), custom_getter=self._rnn_get_variable):
	#		return super(RNNCell, self).__call__(inputs, state)

	def _rnn_get_variable(self, getter, *args, **kwargs):
		variable = getter(*args, **kwargs)
		trainable = (variable in tf_variables.trainable_variables() or
								 (isinstance(variable, tf_variables.PartitionedVariable) and
									list(variable)[0] in tf_variables.trainable_variables()))
		if trainable and variable not in self._trainable_weights:
			self._trainable_weights.append(variable)
		elif not trainable and variable not in self._non_trainable_weights:
			self._non_trainable_weights.append(variable)
		return variable

	@property
	def state_size(self):
		raise NotImplementedError("Abstract method")

	@property
	def output_size(self):
		"""Integer or TensorShape: size of outputs produced by this cell."""
		raise NotImplementedError("Abstract method")

	def build(self, _):
		# This tells the parent Layer object that it's OK to call
		# self.add_variable() inside the call() method.
		pass

	def zero_state(self, batch_size, dtype):
		with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
			state_size = self.state_size
			return _zero_state_tensors(state_size, batch_size, dtype)


_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h"))


class LSTMStateTuple(_LSTMStateTuple):
	"""Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.
	Stores two elements: `(c, h)`, in that order.
	Only used when `state_is_tuple=True`.
	"""
	__slots__ = ()

	@property
	def dtype(self):
		(c, h) = self
		if c.dtype != h.dtype:
			raise TypeError("Inconsistent internal state: %s tf.%s" %
											(str(c.dtype), str(h.dtype)))
		return c.dtype


class CustomLSTMCell(RNNCell):

	def __init__(self, num_units, forget_bias=1.0,
							 state_is_tuple=True, activation=None, reuse=None):

		super(CustomLSTMCell, self).__init__(_reuse=reuse)
		if not state_is_tuple:
			logging.warn("%s: Using a concatenated state is slower and will soon be "
									 "deprecated.	Use state_is_tuple=True.", self)
		self._num_units = num_units
		self._forget_bias = forget_bias
		self._state_is_tuple = state_is_tuple
		self._activation = activation or tf.tanh

	@property
	def state_size(self):
		return (LSTMStateTuple(self._num_units, self._num_units)
						if self._state_is_tuple else 2 * self._num_units)

	@property
	def output_size(self):
		return self._num_units

	def call(self, inputs, state, params=None):
		if not params is None:
			assert len(params) == 2
			
		sigmoid = tf.sigmoid
		# Parameters of gates are concatenated into one multiply for efficiency.
		c, h = state
		concat = _linear([inputs, h], 4 * self._num_units, True, params=params)

		# i = input_gate, j = new_input, f = forget_gate, o = output_gate
		i, j, f, o = tf.split(value=concat, num_or_size_splits=4, axis=1)

		new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j))
		new_h = self._activation(new_c) * sigmoid(o)

		new_state = LSTMStateTuple(new_c, new_h)

		return new_h, new_state


def _linear(args,
		output_size,
		bias,
		bias_initializer=None,
		kernel_initializer=None,
		params=None):

	if args is None or (nest.is_sequence(args) and not args):
		raise ValueError("`args` must be specified")
	if not nest.is_sequence(args):
		args = [args]

	# Calculate the total size of arguments on dimension 1.
	total_arg_size = 0
	shapes = [a.get_shape() for a in args]
	for shape in shapes:
		if shape.ndims != 2:
			raise ValueError("linear is expecting 2D arguments: %s" % shapes)
		if shape[1].value is None:
			raise ValueError("linear expects shape[1] to be provided for shape %s, "
											 "but saw %s" % (shape, shape[1]))
		else:
			total_arg_size += shape[1].value

	dtype = [a.dtype for a in args][0]

	# Now the computation.
	scope = tf.get_variable_scope()
	with tf.variable_scope(scope) as outer_scope:
		if params is None:
			weights = tf.get_variable(
					_WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size],
					dtype=dtype,
					initializer=kernel_initializer)
		else:
			weights = params[0]
					
		if len(args) == 1:
			res = tf.matmul(args[0], weights)
		else:
			res = tf.matmul(tf.concat(args, 1), weights)
			
		if not bias:
			return res
			
		with tf.variable_scope(outer_scope) as inner_scope:
			inner_scope.set_partitioner(None)

			if params is None:
				if bias_initializer is None:
					bias_initializer = tf.constant_initializer(0.0, dtype=dtype)
				biases = tf.get_variable(
						_BIAS_VARIABLE_NAME, [output_size],
						dtype=dtype,
						initializer=bias_initializer)
			else:
				biases = params[1]
		return nn_ops.bias_add(res, biases)
		
		
class CustomMultiRNNCell(RNNCell):
	"""RNN cell composed sequentially of multiple simple cells."""

	def __init__(self, cells, state_is_tuple=True):
		super(CustomMultiRNNCell, self).__init__()
		if not cells:
			raise ValueError("Must specify at least one cell for MultiRNNCell.")
		if not nest.is_sequence(cells):
			raise TypeError(
					"cells must be a list or tuple, but saw: %s." % cells)

		self._cells = cells
		self._state_is_tuple = state_is_tuple
		if not state_is_tuple:
			if any(nest.is_sequence(c.state_size) for c in self._cells):
				raise ValueError("Some cells return tuples of states, but the flag "
												 "state_is_tuple is not set.	State sizes are: %s"
												 % str([c.state_size for c in self._cells]))

	@property
	def state_size(self):
		if self._state_is_tuple:
			return tuple(cell.state_size for cell in self._cells)
		else:
			return sum([cell.state_size for cell in self._cells])

	@property
	def output_size(self):
		return self._cells[-1].output_size

	def zero_state(self, batch_size, dtype):
		with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
			if self._state_is_tuple:
				return tuple(cell.zero_state(batch_size, dtype) for cell in self._cells)
			else:
				# We know here that state_size of each cell is not a tuple and
				# presumably does not contain TensorArrays or anything else fancy
				return super(MultiRNNCell, self).zero_state(batch_size, dtype)

	def call(self, inputs, state, params):
		"""Run this multi-layer cell on inputs, starting from state."""
		cur_state_pos = 0
		cur_inp = inputs
		new_states = []
		for i, cell in enumerate(self._cells):
			with tf.variable_scope("cell_%d" % i):
				if self._state_is_tuple:
					if not nest.is_sequence(state):
						raise ValueError(
								"Expected state to be a tuple of length %d, but received: %s" %
								(len(self.state_size), state))
					cur_state = state[i]
				else:
					cur_state = tf.slice(state, [0, cur_state_pos], [-1, cell.state_size])
					cur_state_pos += cell.state_size
				cur_inp, new_state = cell(cur_inp, cur_state, params[i])
				new_states.append(new_state)

		new_states = (tuple(new_states) if self._state_is_tuple else
									tf.concat(new_states, 1))

		return cur_inp, new_states
			


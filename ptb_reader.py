from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys

import tensorflow as tf


def _read_words(filename):
	with open(filename, "r") as f:
		if sys.version_info[0] >= 3:
			return f.read().replace("\n", "<eos>").split()
		else:
			return f.read().decode("utf-8").replace("\n", "<eos>").split()


def _build_vocab(filename):
	data = _read_words(filename)

	counter = collections.Counter(data)
	count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

	words, _ = list(zip(*count_pairs))
	word_to_id = dict(zip(words, range(len(words))))

	return word_to_id


def _file_to_word_ids(filename, word_to_id):
	data = _read_words(filename)
	return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path = "data/"):
	train_path = os.path.join(data_path, "ptb.train.txt")
	valid_path = os.path.join(data_path, "ptb.valid.txt")
	test_path = os.path.join(data_path, "ptb.test.txt")

	word_to_id = _build_vocab(train_path)
	train_data = _file_to_word_ids(train_path, word_to_id)
	valid_data = _file_to_word_ids(valid_path, word_to_id)
	test_data = _file_to_word_ids(test_path, word_to_id)
	
	return train_data, valid_data, test_data, word_to_id


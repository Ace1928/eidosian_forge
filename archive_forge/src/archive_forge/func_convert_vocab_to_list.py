import copy
import functools
import re
import weakref
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import initializers
from keras.src.utils import io_utils
from tensorflow.python.util.tf_export import keras_export
def convert_vocab_to_list(vocab):
    """Convert input vacabulary to list."""
    vocab_list = []
    if tf.is_tensor(vocab):
        vocab_list = list(vocab.numpy())
    elif isinstance(vocab, (np.ndarray, tuple, list)):
        vocab_list = list(vocab)
    elif isinstance(vocab, str):
        if not tf.io.gfile.exists(vocab):
            raise ValueError(f'Vocabulary file {vocab} does not exist.')
        with tf.io.gfile.GFile(vocab, 'r') as vocabulary_file:
            vocab_list = vocabulary_file.read().splitlines()
    else:
        raise ValueError(f'Vocabulary is expected to be either a NumPy array, list, 1D tensor or a vocabulary text file. Instead type {type(vocab)} was received.')
    if len(vocab_list) == 0:
        raise ValueError('Vocabulary is expected to be either a NumPy array, list, 1D tensor or a vocabulary text file with at least one token. Received 0 instead.')
    return vocab_list
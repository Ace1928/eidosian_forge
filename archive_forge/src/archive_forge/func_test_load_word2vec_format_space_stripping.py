import functools
import logging
import unittest
import numpy as np
from gensim.models.keyedvectors import KeyedVectors, REAL, pseudorandom_weak_vector
from gensim.test.utils import datapath
import gensim.models.keyedvectors
def test_load_word2vec_format_space_stripping(self):
    w2v_dict = {'\nabc': [1, 2, 3], 'cdefdg': [4, 5, 6], '\n\ndef': [7, 8, 9]}
    self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=5, limit=None)
    self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=5, limit=1)
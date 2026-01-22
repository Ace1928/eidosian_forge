import functools
import logging
import unittest
import numpy as np
from gensim.models.keyedvectors import KeyedVectors, REAL, pseudorandom_weak_vector
from gensim.test.utils import datapath
import gensim.models.keyedvectors
def test_add_type(self):
    kv = KeyedVectors(2)
    assert kv.vectors.dtype == REAL
    words, vectors = (['a'], np.array([1.0, 1.0], dtype=np.float64).reshape(1, -1))
    kv.add_vectors(words, vectors)
    assert kv.vectors.dtype == REAL
import logging
import unittest
import numpy as np
from gensim.test.utils import datapath
from gensim.models.keyedvectors import KeyedVectors
def test_type_conversion(self):
    path = datapath('high_precision.kv.txt')
    binary_path = datapath('high_precision.kv.bin')
    model1 = KeyedVectors.load_word2vec_format(path, datatype=np.float16)
    model1.save_word2vec_format(binary_path, binary=True)
    model2 = KeyedVectors.load_word2vec_format(binary_path, datatype=np.float64, binary=True)
    self.assertAlmostEqual(model1['horse.n.01'][0], np.float16(model2['horse.n.01'][0]))
    self.assertEqual(model1['horse.n.01'][0].dtype, np.float16)
    self.assertEqual(model2['horse.n.01'][0].dtype, np.float64)
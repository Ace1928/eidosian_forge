from __future__ import division
import gzip
import io
import logging
import unittest
import os
import shutil
import subprocess
import struct
import sys
import numpy as np
import pytest
from gensim import utils
from gensim.models.word2vec import LineSentence
from gensim.models.fasttext import FastText as FT_gensim, FastTextKeyedVectors, _unpack
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import (
from gensim.test.test_word2vec import TestWord2VecModel
import gensim.models._fasttext_bin
from gensim.models.fasttext_inner import compute_ngrams, compute_ngrams_bytes, ft_hash_bytes
import gensim.models.fasttext
def test_load_fasttext_new_format(self):
    try:
        new_model = gensim.models.fasttext.load_facebook_model(self.test_new_model_file)
    except Exception as exc:
        self.fail('Unable to load FastText model from file %s: %s' % (self.test_new_model_file, exc))
    vocab_size, model_size = (1763, 10)
    self.assertEqual(new_model.wv.vectors.shape, (vocab_size, model_size))
    self.assertEqual(len(new_model.wv), vocab_size, model_size)
    self.assertEqual(new_model.wv.vectors_ngrams.shape, (new_model.wv.bucket, model_size))
    expected_vec = [-0.025627, -0.11448, 0.18116, -0.96779, 0.2532, -0.93224, 0.3929, 0.12679, -0.19685, -0.13179]
    actual_vec = new_model.wv['hundred']
    self.assertTrue(np.allclose(actual_vec, expected_vec, atol=0.0001))
    expected_vec_oov = [-0.49111, -0.13122, -0.02109, -0.88769, -0.20105, -0.91732, 0.47243, 0.19708, -0.17856, 0.19815]
    actual_vec_oov = new_model.wv['rejection']
    self.assertTrue(np.allclose(actual_vec_oov, expected_vec_oov, atol=0.0001))
    self.assertEqual(new_model.min_count, 5)
    self.assertEqual(new_model.window, 5)
    self.assertEqual(new_model.epochs, 5)
    self.assertEqual(new_model.negative, 5)
    self.assertEqual(new_model.sample, 0.0001)
    self.assertEqual(new_model.wv.bucket, 1000)
    self.assertEqual(new_model.wv.max_n, 6)
    self.assertEqual(new_model.wv.min_n, 3)
    self.assertEqual(new_model.wv.vectors.shape, (len(new_model.wv), new_model.vector_size))
    self.assertEqual(new_model.wv.vectors_ngrams.shape, (new_model.wv.bucket, new_model.vector_size))
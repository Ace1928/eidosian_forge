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
def model_structural_sanity(self, model):
    """Check a model for basic self-consistency, necessary properties & property
        correspondences, but no semantic tests."""
    self.assertEqual(model.wv.vectors.shape, (len(model.wv), model.vector_size))
    self.assertEqual(model.wv.vectors_vocab.shape, (len(model.wv), model.vector_size))
    self.assertEqual(model.wv.vectors_ngrams.shape, (model.wv.bucket, model.vector_size))
    self.assertLessEqual(len(model.wv.vectors_ngrams_lockf), len(model.wv.vectors_ngrams))
    self.assertLessEqual(len(model.wv.vectors_vocab_lockf), len(model.wv.index_to_key))
    self.assertTrue(np.isfinite(model.wv.vectors_ngrams).all(), 'NaN in ngrams')
    self.assertTrue(np.isfinite(model.wv.vectors_vocab).all(), 'NaN in vectors_vocab')
    if model.negative:
        self.assertTrue(np.isfinite(model.syn1neg).all(), 'NaN in syn1neg')
    if model.hs:
        self.assertTrue(np.isfinite(model.syn1).all(), 'NaN in syn1neg')
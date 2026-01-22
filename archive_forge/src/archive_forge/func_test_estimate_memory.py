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
def test_estimate_memory(self):
    model = FT_gensim(sg=1, hs=1, vector_size=12, negative=5, min_count=3, bucket=BUCKET)
    model.build_vocab(sentences)
    report = model.estimate_memory()
    self.assertEqual(report['vocab'], 2800)
    self.assertEqual(report['syn0_vocab'], 192)
    self.assertEqual(report['syn1'], 192)
    self.assertEqual(report['syn1neg'], 192)
    self.assertEqual(report['syn0_ngrams'], model.vector_size * np.dtype(np.float32).itemsize * BUCKET)
    self.assertEqual(report['buckets_word'], 688)
    self.assertEqual(report['total'], 484064)
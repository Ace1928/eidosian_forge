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
def test_load_native_pretrained(self):
    model = gensim.models.fasttext.load_facebook_model(datapath('toy-model-pretrained.bin'))
    actual = model.wv['monarchist']
    expected = np.array([0.76222, 1.0669, 0.7055, -0.090969, -0.53508])
    self.assertTrue(np.allclose(expected, actual, atol=0.001))
    self.model_structural_sanity(model)
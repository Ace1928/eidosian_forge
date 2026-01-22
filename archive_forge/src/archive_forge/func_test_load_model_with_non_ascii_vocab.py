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
def test_load_model_with_non_ascii_vocab(self):
    model = gensim.models.fasttext.load_facebook_model(datapath('non_ascii_fasttext.bin'))
    self.assertTrue(u'který' in model.wv)
    try:
        model.wv[u'který']
    except UnicodeDecodeError:
        self.fail('Unable to access vector for utf8 encoded non-ascii word')
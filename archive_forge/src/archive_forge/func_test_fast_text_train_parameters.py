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
def test_fast_text_train_parameters(self):
    model = FT_gensim(vector_size=12, min_count=1, hs=1, negative=0, seed=42, workers=1, bucket=BUCKET)
    model.build_vocab(corpus_iterable=sentences)
    self.assertRaises(TypeError, model.train, corpus_file=11111, total_examples=1, epochs=1)
    self.assertRaises(TypeError, model.train, corpus_iterable=11111, total_examples=1, epochs=1)
    self.assertRaises(TypeError, model.train, corpus_iterable=sentences, corpus_file='test', total_examples=1, epochs=1)
    self.assertRaises(TypeError, model.train, corpus_iterable=None, corpus_file=None, total_examples=1, epochs=1)
    self.assertRaises(TypeError, model.train, corpus_file=sentences, total_examples=1, epochs=1)
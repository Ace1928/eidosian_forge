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
def compare_vocabulary(a, b, t):
    t.assertEqual(a.max_vocab_size, b.max_vocab_size)
    t.assertEqual(a.min_count, b.min_count)
    t.assertEqual(a.sample, b.sample)
    t.assertEqual(a.sorted_vocab, b.sorted_vocab)
    t.assertEqual(a.null_word, b.null_word)
    t.assertTrue(np.allclose(a.cum_table, b.cum_table))
    t.assertEqual(a.raw_vocab, b.raw_vocab)
    t.assertEqual(a.max_final_vocab, b.max_final_vocab)
    t.assertEqual(a.ns_exponent, b.ns_exponent)
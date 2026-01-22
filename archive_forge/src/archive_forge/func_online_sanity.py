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
def online_sanity(self, model):
    terro, others = ([], [])
    for line in list_corpus:
        if 'terrorism' in line:
            terro.append(line)
        else:
            others.append(line)
    self.assertTrue(all(('terrorism' not in line for line in others)))
    model.build_vocab(others)
    start_vecs = model.wv.vectors_vocab.copy()
    model.train(others, total_examples=model.corpus_count, epochs=model.epochs)
    self.assertFalse(np.all(np.equal(start_vecs, model.wv.vectors_vocab)))
    self.assertFalse(np.all(np.equal(model.wv.vectors, model.wv.vectors_vocab)))
    self.assertFalse('terrorism' in model.wv.key_to_index)
    model.build_vocab(terro, update=True)
    self.assertTrue(model.wv.vectors_ngrams.dtype == 'float32')
    self.assertTrue('terrorism' in model.wv.key_to_index)
    orig0_all = np.copy(model.wv.vectors_ngrams)
    model.train(terro, total_examples=len(terro), epochs=model.epochs)
    self.assertFalse(np.allclose(model.wv.vectors_ngrams, orig0_all))
    sim = model.wv.n_similarity(['war'], ['terrorism'])
    assert abs(sim) > 0.6
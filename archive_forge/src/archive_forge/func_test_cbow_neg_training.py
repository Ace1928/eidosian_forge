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
def test_cbow_neg_training(self):
    model_gensim = FT_gensim(vector_size=48, sg=0, cbow_mean=1, alpha=0.05, window=5, hs=0, negative=5, min_count=5, epochs=10, batch_words=1000, word_ngrams=1, sample=0.001, min_n=3, max_n=6, sorted_vocab=1, workers=1, min_alpha=0.0, bucket=BUCKET)
    lee_data = LineSentence(datapath('lee_background.cor'))
    model_gensim.build_vocab(lee_data)
    orig0 = np.copy(model_gensim.wv.vectors[0])
    model_gensim.train(lee_data, total_examples=model_gensim.corpus_count, epochs=model_gensim.epochs)
    self.assertFalse((orig0 == model_gensim.wv.vectors[0]).all())
    sims_gensim = model_gensim.wv.most_similar('night', topn=10)
    sims_gensim_words = [word for word, distance in sims_gensim]
    expected_sims_words = [u'night.', u'night,', u'eight', u'fight', u'month', u'hearings', u'Washington', u'remains', u'overnight', u'running']
    overlaps = set(sims_gensim_words).intersection(expected_sims_words)
    overlap_count = len(overlaps)
    self.assertGreaterEqual(overlap_count, 2, 'only %i overlap in expected %s & actual %s' % (overlap_count, expected_sims_words, sims_gensim_words))
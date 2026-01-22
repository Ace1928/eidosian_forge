import logging
import unittest
import os
import bz2
import sys
import tempfile
import subprocess
import numpy as np
from testfixtures import log_capture
from gensim import utils
from gensim.models import word2vec, keyedvectors
from gensim.utils import check_output
from gensim.test.utils import (
def onlineSanity(self, model, trained_model=False):
    terro, others = ([], [])
    for line in lee_corpus_list:
        if 'terrorism' in line:
            terro.append(line)
        else:
            others.append(line)
    self.assertTrue(all(('terrorism' not in line for line in others)))
    model.build_vocab(others, update=trained_model)
    model.train(others, total_examples=model.corpus_count, epochs=model.epochs)
    self.assertFalse('terrorism' in model.wv)
    model.build_vocab(terro, update=True)
    self.assertTrue('terrorism' in model.wv)
    orig0 = np.copy(model.wv.vectors)
    model.train(terro, total_examples=len(terro), epochs=model.epochs)
    self.assertFalse(np.allclose(model.wv.vectors, orig0))
    sim = model.wv.n_similarity(['war'], ['terrorism'])
    self.assertLess(0.0, sim)
from __future__ import with_statement, division
import logging
import unittest
import os
from collections import namedtuple
import numpy as np
from testfixtures import log_capture
from gensim import utils
from gensim.models import doc2vec, keyedvectors
from gensim.test.utils import datapath, get_tmpfile, temporary_file, common_texts as raw_sentences
def models_equal(self, model, model2):
    self.assertEqual(len(model.wv), len(model2.wv))
    self.assertTrue(np.allclose(model.wv.vectors, model2.wv.vectors))
    if model.hs:
        self.assertTrue(np.allclose(model.syn1, model2.syn1))
    if model.negative:
        self.assertTrue(np.allclose(model.syn1neg, model2.syn1neg))
    self.assertEqual(len(model.dv), len(model2.dv))
    self.assertEqual(len(model.dv.index_to_key), len(model2.dv.index_to_key))
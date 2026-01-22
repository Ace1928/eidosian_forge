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
def obsolete_testLoadOldModelSeparates(self):
    """Test loading an old doc2vec model from indeterminate version"""
    model_file = 'doc2vec_old_sep'
    model = doc2vec.Doc2Vec.load(datapath(model_file))
    self.assertTrue(model.wv.vectors.shape == (3955, 100))
    self.assertTrue(len(model.wv) == 3955)
    self.assertTrue(len(model.wv.index_to_key) == 3955)
    self.assertIsNone(model.corpus_total_words)
    self.assertTrue(model.syn1neg.shape == (len(model.wv), model.vector_size))
    self.assertTrue(model.wv.vectors_lockf.shape == (3955,))
    self.assertTrue(model.cum_table.shape == (3955,))
    self.assertTrue(model.dv.vectors.shape == (300, 100))
    self.assertTrue(model.dv.vectors_lockf.shape == (300,))
    self.assertTrue(len(model.dv) == 300)
    self.model_sanity(model)
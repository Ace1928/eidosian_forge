from __future__ import with_statement
import logging
import unittest
from functools import partial
import numpy as np
from gensim import corpora, models, utils, matutils
from gensim.parsing.preprocessing import preprocess_documents, preprocess_string, DEFAULT_FILTERS
from gensim.test.utils import datapath
def test_corpus(self):
    """availability and integrity of corpus"""
    documents_in_bg_corpus = 300
    documents_in_corpus = 50
    len_sim_vector = 1225
    self.assertEqual(len(bg_corpus), documents_in_bg_corpus)
    self.assertEqual(len(corpus), documents_in_corpus)
    self.assertEqual(len(human_sim_vector), len_sim_vector)
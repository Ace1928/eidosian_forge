import logging
import unittest
import numpy as np
from gensim.corpora.mmcorpus import MmCorpus
from gensim.models import tfidfmodel
from gensim.test.utils import datapath, get_tmpfile, common_dictionary, common_corpus
from gensim.corpora import Dictionary
def test_pivoted_normalization(self):
    docs = [corpus[1], corpus[2]]
    model = tfidfmodel.TfidfModel(self.corpus)
    transformed_docs = [model[docs[0]], model[docs[1]]]
    model = tfidfmodel.TfidfModel(self.corpus, pivot=0, slope=1)
    expected_docs = [model[docs[0]], model[docs[1]]]
    self.assertTrue(np.allclose(sorted(transformed_docs[0]), sorted(expected_docs[0])))
    self.assertTrue(np.allclose(sorted(transformed_docs[1]), sorted(expected_docs[1])))
    model = tfidfmodel.TfidfModel(self.corpus, pivot=0, slope=0.5)
    transformed_docs = [model[docs[0]], model[docs[1]]]
    expected_docs = [[(8, 0.8884910505493495), (7, 0.648974041227711), (6, 0.8884910505493495), (5, 0.648974041227711), (4, 0.8884910505493495), (3, 0.8884910505493495)], [(10, 0.8164965809277263), (9, 0.8164965809277263), (5, 1.6329931618554525)]]
    self.assertTrue(np.allclose(sorted(transformed_docs[0]), sorted(expected_docs[0])))
    self.assertTrue(np.allclose(sorted(transformed_docs[1]), sorted(expected_docs[1])))
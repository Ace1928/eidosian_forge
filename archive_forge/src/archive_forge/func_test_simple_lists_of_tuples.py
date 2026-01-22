import logging
import unittest
import numpy as np
from gensim import utils
from gensim.test.utils import datapath, get_tmpfile
def test_simple_lists_of_tuples(self):
    potentialCorpus = [[(0, 4.0)]]
    result = utils.is_corpus(potentialCorpus)
    expected = (True, potentialCorpus)
    self.assertEqual(expected, result)
    potentialCorpus = [[(0, 4.0), (1, 2.0)]]
    result = utils.is_corpus(potentialCorpus)
    expected = (True, potentialCorpus)
    self.assertEqual(expected, result)
    potentialCorpus = [[(0, 4.0), (1, 2.0), (2, 5.0), (3, 8.0)]]
    result = utils.is_corpus(potentialCorpus)
    expected = (True, potentialCorpus)
    self.assertEqual(expected, result)
    potentialCorpus = [[(0, 4.0)], [(1, 2.0)]]
    result = utils.is_corpus(potentialCorpus)
    expected = (True, potentialCorpus)
    self.assertEqual(expected, result)
    potentialCorpus = [[(0, 4.0)], [(1, 2.0)], [(2, 5.0)], [(3, 8.0)]]
    result = utils.is_corpus(potentialCorpus)
    expected = (True, potentialCorpus)
    self.assertEqual(expected, result)
from collections.abc import Mapping
from itertools import chain
import logging
import unittest
import codecs
import os
import os.path
import scipy
import gensim
from gensim.corpora import Dictionary
from gensim.utils import to_utf8
from gensim.test.utils import get_tmpfile, common_texts
def test_filter_tokens(self):
    self.maxDiff = 10000
    d = Dictionary(self.texts)
    removed_word = d[0]
    d.filter_tokens([0])
    expected = {'computer': 0, 'eps': 8, 'graph': 10, 'human': 1, 'interface': 2, 'minors': 11, 'response': 3, 'survey': 4, 'system': 5, 'time': 6, 'trees': 9, 'user': 7}
    del expected[removed_word]
    self.assertEqual(sorted(d.token2id.keys()), sorted(expected.keys()))
    expected[removed_word] = len(expected)
    d.add_documents([[removed_word]])
    self.assertEqual(sorted(d.token2id.keys()), sorted(expected.keys()))
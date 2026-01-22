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
def test_loadFromText(self):
    """`Dictionary` can be loaded from textfile."""
    tmpf = get_tmpfile('load_dict_test.txt')
    no_num_docs_serialization = to_utf8('2\n1\tprvé\t1\n2\tslovo\t2\n')
    with open(tmpf, 'wb') as file:
        file.write(no_num_docs_serialization)
    d = Dictionary.load_from_text(tmpf)
    self.assertEqual(d.token2id[u'prvé'], 1)
    self.assertEqual(d.token2id[u'slovo'], 2)
    self.assertEqual(d.dfs[1], 1)
    self.assertEqual(d.dfs[2], 2)
    self.assertEqual(d.num_docs, 2)
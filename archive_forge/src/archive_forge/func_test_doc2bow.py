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
def test_doc2bow(self):
    d = Dictionary([['žluťoučký'], ['žluťoučký']])
    self.assertEqual(d.doc2bow(['žluťoučký']), [(0, 1)])
    self.assertRaises(TypeError, d.doc2bow, 'žluťoučký')
    self.assertEqual(d.doc2bow([u'žluťoučký']), [(0, 1)])
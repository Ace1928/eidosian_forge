from __future__ import unicode_literals
import codecs
import itertools
import logging
import os
import os.path
import tempfile
import unittest
import numpy as np
from gensim.corpora import (bleicorpus, mmcorpus, lowcorpus, svmlightcorpus,
from gensim.interfaces import TransformedCorpus
from gensim.utils import to_unicode
from gensim.test.utils import datapath, get_tmpfile, common_corpus
def test_two_level_directory(self):
    dirpath, next_level = self.write_two_levels()
    corpus = textcorpus.TextDirectoryCorpus(dirpath)
    self.assertEqual(len(corpus), 4)
    docs = list(corpus)
    self.assertEqual(len(docs), 4)
    corpus = textcorpus.TextDirectoryCorpus(dirpath, min_depth=1)
    self.assertEqual(len(corpus), 2)
    docs = list(corpus)
    self.assertEqual(len(docs), 2)
    corpus = textcorpus.TextDirectoryCorpus(dirpath, max_depth=0)
    self.assertEqual(len(corpus), 2)
    docs = list(corpus)
    self.assertEqual(len(docs), 2)
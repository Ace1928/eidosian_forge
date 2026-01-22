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
def test_first_element(self):
    """
        First two articles in this sample are
        1) anarchism
        2) autism
        """
    corpus = self.corpus_class(self.enwiki, processes=1)
    texts = corpus.get_texts()
    self.assertTrue(u'anarchism' in next(texts))
    self.assertTrue(u'autism' in next(texts))
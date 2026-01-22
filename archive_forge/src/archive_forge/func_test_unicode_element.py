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
def test_unicode_element(self):
    """
        First unicode article in this sample is
        1) папа
        """
    bgwiki = datapath('bgwiki-latest-pages-articles-shortened.xml.bz2')
    corpus = self.corpus_class(bgwiki)
    texts = corpus.get_texts()
    self.assertTrue(u'папа' in next(texts))
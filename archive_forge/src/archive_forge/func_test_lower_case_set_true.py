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
def test_lower_case_set_true(self):
    """
        Set the parameter lower to True and check that upper case 'Anarchism' token doesnt exist
        """
    corpus = self.corpus_class(self.enwiki, processes=1, lower=True)
    row = corpus.get_texts()
    list_tokens = next(row)
    self.assertTrue(u'Anarchism' not in list_tokens)
    self.assertTrue(u'anarchism' in list_tokens)
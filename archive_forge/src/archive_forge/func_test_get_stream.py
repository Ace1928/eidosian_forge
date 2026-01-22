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
def test_get_stream(self):
    wiki = self.corpus_class(self.enwiki)
    sample_text_wiki = next(wiki.getstream()).decode()[1:14]
    self.assertEqual(sample_text_wiki, 'mediawiki xml')
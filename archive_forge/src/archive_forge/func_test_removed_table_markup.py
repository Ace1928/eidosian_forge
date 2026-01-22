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
def test_removed_table_markup(self):
    """
        Check if all the table markup has been removed.
        """
    enwiki_file = datapath('enwiki-table-markup.xml.bz2')
    corpus = self.corpus_class(enwiki_file)
    texts = corpus.get_texts()
    table_markup = ['style', 'class', 'border', 'cellspacing', 'cellpadding', 'colspan', 'rowspan']
    for text in texts:
        for word in table_markup:
            self.assertTrue(word not in text)
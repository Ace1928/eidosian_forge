from __future__ import division  # always use floats
from __future__ import with_statement
import logging
import os
import unittest
from gensim import utils, corpora, models, similarities
from gensim.test.utils import datapath, get_tmpfile
def test_textcorpus(self):
    """Make sure TextCorpus can be serialized to disk. """
    miislita = CorpusMiislita(datapath('head500.noblanks.cor.bz2'))
    ftmp = get_tmpfile('test_textcorpus.mm')
    corpora.MmCorpus.save_corpus(ftmp, miislita)
    self.assertTrue(os.path.exists(ftmp))
    miislita2 = corpora.MmCorpus(ftmp)
    self.assertEqual(list(miislita), list(miislita2))
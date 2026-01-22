from __future__ import division  # always use floats
from __future__ import with_statement
import logging
import os
import unittest
from gensim import utils, corpora, models, similarities
from gensim.test.utils import datapath, get_tmpfile
def test_save_load_ability(self):
    """
        Make sure we can save and load (un/pickle) TextCorpus objects (as long
        as the underlying input isn't a file-like object; we cannot pickle those).
        """
    corpusname = datapath('miIslita.cor')
    miislita = CorpusMiislita(corpusname)
    tmpf = get_tmpfile('tc_test.cpickle')
    miislita.save(tmpf)
    miislita2 = CorpusMiislita.load(tmpf)
    self.assertEqual(len(miislita), len(miislita2))
    self.assertEqual(miislita.dictionary.token2id, miislita2.dictionary.token2id)
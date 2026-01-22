import logging
import unittest
import numpy as np
import scipy.linalg
from gensim import matutils
from gensim.corpora.mmcorpus import MmCorpus
from gensim.models import lsimodel
from gensim.test import basetmtests
from gensim.test.utils import datapath, get_tmpfile
def test_large_mmap(self):
    fname = get_tmpfile('gensim_models_lsi.tst')
    model = self.model
    model.save(fname, sep_limit=0)
    model2 = lsimodel.LsiModel.load(fname, mmap='r')
    self.assertEqual(model.num_topics, model2.num_topics)
    self.assertTrue(isinstance(model2.projection.u, np.memmap))
    self.assertTrue(isinstance(model2.projection.s, np.memmap))
    self.assertTrue(np.allclose(model.projection.u, model2.projection.u))
    self.assertTrue(np.allclose(model.projection.s, model2.projection.s))
    tstvec = []
    self.assertTrue(np.allclose(model[tstvec], model2[tstvec]))
import logging
import unittest
import numpy as np
from gensim.corpora.mmcorpus import MmCorpus
from gensim.models import logentropy_model
from gensim.test.utils import datapath, get_tmpfile
def test_persistence_compressed(self):
    fname = get_tmpfile('gensim_models_logentry.tst.gz')
    model = logentropy_model.LogEntropyModel(self.corpus_ok, normalize=True)
    model.save(fname)
    model2 = logentropy_model.LogEntropyModel.load(fname, mmap=None)
    self.assertTrue(model.entr == model2.entr)
    tstvec = []
    self.assertTrue(np.allclose(model[tstvec], model2[tstvec]))
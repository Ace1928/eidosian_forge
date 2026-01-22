import logging
import numbers
import os
import unittest
import copy
import numpy as np
from numpy.testing import assert_allclose
from gensim.corpora import mmcorpus, Dictionary
from gensim.models import ldamodel, ldamulticore
from gensim import matutils, utils
from gensim.test import basetmtests
from gensim.test.utils import datapath, get_tmpfile, common_texts
def test_persistence_ignore(self):
    fname = get_tmpfile('gensim_models_lda_testPersistenceIgnore.tst')
    model = ldamodel.LdaModel(self.corpus, num_topics=2)
    model.save(fname, ignore='id2word')
    model2 = ldamodel.LdaModel.load(fname)
    self.assertTrue(model2.id2word is None)
    model.save(fname, ignore=['id2word'])
    model2 = ldamodel.LdaModel.load(fname)
    self.assertTrue(model2.id2word is None)
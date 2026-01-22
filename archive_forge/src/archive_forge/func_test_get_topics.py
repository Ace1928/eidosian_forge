import logging
import unittest
import numpy as np
import scipy.linalg
from gensim import matutils
from gensim.corpora.mmcorpus import MmCorpus
from gensim.models import lsimodel
from gensim.test import basetmtests
from gensim.test.utils import datapath, get_tmpfile
def test_get_topics(self):
    topics = self.model.get_topics()
    vocab_size = len(self.model.id2word)
    for topic in topics:
        self.assertTrue(isinstance(topic, np.ndarray))
        self.assertEqual(topic.dtype, np.float64)
        self.assertEqual(vocab_size, topic.shape[0])
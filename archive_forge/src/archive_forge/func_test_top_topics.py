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
def test_top_topics(self):
    top_topics = self.model.top_topics(self.corpus)
    for topic, score in top_topics:
        self.assertTrue(isinstance(topic, list))
        self.assertTrue(isinstance(score, float))
        for v, k in topic:
            self.assertTrue(isinstance(k, str))
            self.assertTrue(np.issubdtype(v, np.floating))
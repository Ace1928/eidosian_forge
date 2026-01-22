import logging
import unittest
import os
import bz2
import sys
import tempfile
import subprocess
import numpy as np
from testfixtures import log_capture
from gensim import utils
from gensim.models import word2vec, keyedvectors
from gensim.utils import check_output
from gensim.test.utils import (
def test_cosmul(self):
    model = word2vec.Word2Vec(sentences, vector_size=2, min_count=1, hs=1, negative=0)
    sims = model.wv.most_similar_cosmul('graph', topn=10)
    graph_vector = model.wv.get_vector('graph', norm=True)
    sims2 = model.wv.most_similar_cosmul(positive=[graph_vector], topn=11)
    sims2 = [(w, sim) for w, sim in sims2 if w != 'graph']
    self.assertEqual(sims, sims2)
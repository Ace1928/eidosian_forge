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
def test_cbow_hs_fromfile(self):
    model = word2vec.Word2Vec(sg=0, cbow_mean=1, alpha=0.1, window=2, hs=1, negative=0, min_count=5, epochs=60, workers=2, batch_words=1000)
    self.model_sanity(model, with_corpus_file=True)
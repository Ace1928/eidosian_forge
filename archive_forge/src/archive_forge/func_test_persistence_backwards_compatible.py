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
def test_persistence_backwards_compatible(self):
    """Can we still load a model created with an older gensim version?"""
    path = datapath('model-from-gensim-3.8.0.w2v')
    model = word2vec.Word2Vec.load(path)
    x = model.score(['test'])
    assert x is not None
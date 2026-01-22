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
def test_load_pre_keyed_vector_model_c_format(self):
    """Test loading pre-KeyedVectors word2vec model saved in word2vec format"""
    model = keyedvectors.KeyedVectors.load_word2vec_format(datapath('word2vec_pre_kv_c'))
    self.assertTrue(model.vectors.shape[0] == len(model))
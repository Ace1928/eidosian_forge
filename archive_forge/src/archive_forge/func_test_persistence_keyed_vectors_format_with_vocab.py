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
def test_persistence_keyed_vectors_format_with_vocab(self):
    """Test storing/loading the entire model and vocabulary in word2vec format."""
    tmpf = get_tmpfile('gensim_word2vec.tst')
    model = word2vec.Word2Vec(sentences, min_count=1)
    testvocab = get_tmpfile('gensim_word2vec.vocab')
    model.wv.save_word2vec_format(tmpf, testvocab, binary=True)
    kv_binary_model_with_vocab = keyedvectors.KeyedVectors.load_word2vec_format(tmpf, testvocab, binary=True)
    self.assertEqual(model.wv.get_vecattr('human', 'count'), kv_binary_model_with_vocab.get_vecattr('human', 'count'))
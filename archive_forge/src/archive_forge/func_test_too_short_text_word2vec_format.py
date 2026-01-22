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
def test_too_short_text_word2vec_format(self):
    tfile = get_tmpfile('gensim_word2vec.tst')
    model = word2vec.Word2Vec(sentences, min_count=1)
    model.wv.save_word2vec_format(tfile, binary=False)
    f = open(tfile, 'r+b')
    f.write(b'13')
    f.close()
    self.assertRaises(EOFError, keyedvectors.KeyedVectors.load_word2vec_format, tfile, binary=False)
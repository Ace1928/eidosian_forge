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
def test_path_line_sentences(self):
    """Does PathLineSentences work with a path argument?"""
    with utils.open(os.path.join(datapath('PathLineSentences'), '1.txt'), 'rb') as orig1:
        with utils.open(os.path.join(datapath('PathLineSentences'), '2.txt.bz2'), 'rb') as orig2:
            sentences = word2vec.PathLineSentences(datapath('PathLineSentences'))
            orig = orig1.readlines() + orig2.readlines()
            orig_counter = 0
            for words in sentences:
                self.assertEqual(words, utils.to_unicode(orig[orig_counter]).split())
                orig_counter += 1
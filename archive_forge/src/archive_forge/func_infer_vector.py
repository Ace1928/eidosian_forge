from __future__ import with_statement, division
import logging
import unittest
import os
from collections import namedtuple
import numpy as np
from testfixtures import log_capture
from gensim import utils
from gensim.models import doc2vec, keyedvectors
from gensim.test.utils import datapath, get_tmpfile, temporary_file, common_texts as raw_sentences
def infer_vector(self, document, alpha=None, min_alpha=None, epochs=None):
    return np.concatenate([model.infer_vector(document, alpha, min_alpha, epochs) for model in self.models])
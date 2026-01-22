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
def test_dbow_fixedwindowsize(self):
    """Test DBOW doc2vec training with fixed window size."""
    model = doc2vec.Doc2Vec(list_corpus, vector_size=16, shrink_windows=False, dm=0, hs=0, negative=5, min_count=2, epochs=20)
    self.model_sanity(model)
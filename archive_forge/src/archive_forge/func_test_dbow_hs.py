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
def test_dbow_hs(self):
    """Test DBOW doc2vec training."""
    model = doc2vec.Doc2Vec(list_corpus, dm=0, hs=1, negative=0, min_count=2, epochs=20)
    self.model_sanity(model)
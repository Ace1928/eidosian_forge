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
def test_load_mmap(self):
    """Test storing/loading the entire model."""
    model = doc2vec.Doc2Vec(sentences, min_count=1)
    tmpf = get_tmpfile('gensim_doc2vec.tst')
    model.save(tmpf, sep_limit=0)
    self.models_equal(model, doc2vec.Doc2Vec.load(tmpf))
    self.models_equal(model, doc2vec.Doc2Vec.load(tmpf, mmap='r'))
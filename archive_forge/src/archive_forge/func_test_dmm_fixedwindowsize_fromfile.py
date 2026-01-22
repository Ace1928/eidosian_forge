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
def test_dmm_fixedwindowsize_fromfile(self):
    """Test DMM doc2vec training with fixed window size, from file."""
    with temporary_file(get_tmpfile('gensim_doc2vec.tst')) as corpus_file:
        save_lee_corpus_as_line_sentence(corpus_file)
        model = doc2vec.Doc2Vec(corpus_file=corpus_file, vector_size=24, dm=1, dm_mean=1, window=4, shrink_windows=False, hs=0, negative=10, alpha=0.05, min_count=2, epochs=20)
        self.model_sanity(model)
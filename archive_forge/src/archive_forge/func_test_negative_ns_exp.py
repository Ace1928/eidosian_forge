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
def test_negative_ns_exp(self):
    """The model should accept a negative ns_exponent as a valid value."""
    model = doc2vec.Doc2Vec(sentences, ns_exponent=-1, min_count=1, workers=1)
    tmpf = get_tmpfile('d2v_negative_exp.tst')
    model.save(tmpf)
    loaded_model = doc2vec.Doc2Vec.load(tmpf)
    loaded_model.train(sentences, total_examples=model.corpus_count, epochs=1)
    assert loaded_model.ns_exponent == -1, loaded_model.ns_exponent
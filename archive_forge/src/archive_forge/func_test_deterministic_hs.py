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
def test_deterministic_hs(self):
    """Test doc2vec results identical with identical RNG seed."""
    model = doc2vec.Doc2Vec(DocsLeeCorpus(), seed=42, workers=1)
    model2 = doc2vec.Doc2Vec(DocsLeeCorpus(), seed=42, workers=1)
    self.models_equal(model, model2)
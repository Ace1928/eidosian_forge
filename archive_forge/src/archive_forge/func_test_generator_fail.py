import logging
import unittest
import numpy as np
from gensim.corpora.mmcorpus import MmCorpus
from gensim.models import logentropy_model
from gensim.test.utils import datapath, get_tmpfile
def test_generator_fail(self):
    """Test creating a model using a generator as input; should fail."""

    def get_generator(test_corpus=TestLogEntropyModel.TEST_CORPUS):
        for test_doc in test_corpus:
            yield test_doc
    self.assertRaises(ValueError, logentropy_model.LogEntropyModel, corpus=get_generator())
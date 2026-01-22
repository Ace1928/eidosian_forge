import logging
import unittest
import numpy as np
from gensim.corpora.mmcorpus import MmCorpus
from gensim.models import logentropy_model
from gensim.test.utils import datapath, get_tmpfile
def test_empty_fail(self):
    """Test creating a model using an empty input; should fail."""
    self.assertRaises(ValueError, logentropy_model.LogEntropyModel, corpus=self.corpus_empty)
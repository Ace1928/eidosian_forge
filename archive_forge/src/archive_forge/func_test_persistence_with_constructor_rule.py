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
def test_persistence_with_constructor_rule(self):
    """Test storing/loading the entire model with a vocab trimming rule passed in the constructor."""
    tmpf = get_tmpfile('gensim_word2vec.tst')
    model = word2vec.Word2Vec(sentences, min_count=1, trim_rule=_rule)
    model.save(tmpf)
    self.models_equal(model, word2vec.Word2Vec.load(tmpf))
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
def test_lambda_rule(self):
    """Test that lambda trim_rule works."""

    def rule(word, count, min_count):
        return utils.RULE_DISCARD if word == 'human' else utils.RULE_DEFAULT
    model = word2vec.Word2Vec(sentences, min_count=1, trim_rule=rule)
    self.assertTrue('human' not in model.wv)
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
def test_compute_training_loss(self):
    model = word2vec.Word2Vec(min_count=1, sg=1, negative=5, hs=1)
    model.build_vocab(sentences)
    model.train(sentences, compute_loss=True, total_examples=model.corpus_count, epochs=model.epochs)
    training_loss_val = model.get_latest_training_loss()
    self.assertTrue(training_loss_val > 0.0)
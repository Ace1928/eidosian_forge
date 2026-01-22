import os
import logging
import unittest
import numpy as np
from copy import deepcopy
import pytest
from gensim.models import EnsembleLda, LdaMulticore, LdaModel
from gensim.test.utils import datapath, get_tmpfile, common_corpus, common_dictionary
def test_not_trained_given_zero_models(self):
    elda = EnsembleLda(corpus=common_corpus, id2word=common_dictionary, num_topics=NUM_TOPICS, passes=PASSES, num_models=0, random_state=RANDOM_STATE)
    assert len(elda.ttda) == 0
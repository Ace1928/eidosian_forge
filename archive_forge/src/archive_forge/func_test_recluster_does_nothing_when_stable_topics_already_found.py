import os
import logging
import unittest
import numpy as np
from copy import deepcopy
import pytest
from gensim.models import EnsembleLda, LdaMulticore, LdaModel
from gensim.test.utils import datapath, get_tmpfile, common_corpus, common_dictionary
def test_recluster_does_nothing_when_stable_topics_already_found(self):
    elda = self.get_elda()
    elda.recluster()
    assert elda.stable_topics.shape[1] == len(common_dictionary)
    assert len(elda.ttda) == NUM_MODELS * NUM_TOPICS
    self.assert_ttda_is_valid(elda)
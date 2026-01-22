import functools
import logging
import unittest
import numpy as np
from gensim.models.keyedvectors import KeyedVectors, REAL, pseudorandom_weak_vector
from gensim.test.utils import datapath
import gensim.models.keyedvectors
def test_relative_cosine_similarity(self):
    """Test relative_cosine_similarity returns expected results with an input of a word pair and topn"""
    wordnet_syn = ['good', 'goodness', 'commodity', 'trade_good', 'full', 'estimable', 'honorable', 'respectable', 'beneficial', 'just', 'upright', 'adept', 'expert', 'practiced', 'proficient', 'skillful', 'skilful', 'dear', 'near', 'dependable', 'safe', 'secure', 'right', 'ripe', 'well', 'effective', 'in_effect', 'in_force', 'serious', 'sound', 'salutary', 'honest', 'undecomposed', 'unspoiled', 'unspoilt', 'thoroughly', 'soundly']
    cos_sim = [self.vectors.similarity('good', syn) for syn in wordnet_syn if syn in self.vectors]
    cos_sim = sorted(cos_sim, reverse=True)
    rcs_wordnet = self.vectors.similarity('good', 'nice') / sum((cos_sim[i] for i in range(10)))
    rcs = self.vectors.relative_cosine_similarity('good', 'nice', 10)
    self.assertTrue(rcs_wordnet >= rcs)
    self.assertTrue(np.allclose(rcs_wordnet, rcs, 0, 0.125))
    rcs = self.vectors.relative_cosine_similarity('good', 'worst', 10)
    self.assertTrue(rcs < 0.1)
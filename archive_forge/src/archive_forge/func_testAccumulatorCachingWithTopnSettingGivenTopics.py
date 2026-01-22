import logging
import unittest
import multiprocessing as mp
from functools import partial
import numpy as np
from gensim.matutils import argsort
from gensim.models.coherencemodel import CoherenceModel, BOOLEAN_DOCUMENT_BASED
from gensim.models.ldamodel import LdaModel
from gensim.test.utils import get_tmpfile, common_texts, common_dictionary, common_corpus
def testAccumulatorCachingWithTopnSettingGivenTopics(self):
    kwargs = dict(corpus=self.corpus, dictionary=self.dictionary, topn=5, coherence='u_mass')
    cm1 = CoherenceModel(topics=self.topics1, **kwargs)
    cm1.estimate_probabilities()
    self.assertIsNotNone(cm1._accumulator)
    accumulator = cm1._accumulator
    topics_before = cm1._topics
    cm1.topn = 3
    self.assertEqual(accumulator, cm1._accumulator)
    self.assertEqual(3, len(cm1.topics[0]))
    self.assertEqual(topics_before, cm1._topics)
    cm1.topn = 4
    self.assertEqual(accumulator, cm1._accumulator)
    self.assertEqual(4, len(cm1.topics[0]))
    self.assertEqual(topics_before, cm1._topics)
    with self.assertRaises(ValueError):
        cm1.topn = 6
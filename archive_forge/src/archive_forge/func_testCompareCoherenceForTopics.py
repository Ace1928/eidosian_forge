import logging
import unittest
import multiprocessing as mp
from functools import partial
import numpy as np
from gensim.matutils import argsort
from gensim.models.coherencemodel import CoherenceModel, BOOLEAN_DOCUMENT_BASED
from gensim.models.ldamodel import LdaModel
from gensim.test.utils import get_tmpfile, common_texts, common_dictionary, common_corpus
def testCompareCoherenceForTopics(self):
    topics = [self.topics1, self.topics2]
    cm = CoherenceModel.for_topics(topics, dictionary=self.dictionary, texts=self.texts, coherence='c_v')
    self.assertIsNotNone(cm._accumulator)
    for topic_list in topics:
        cm.topics = topic_list
        self.assertIsNotNone(cm._accumulator)
    (coherence_topics1, coherence1), (coherence_topics2, coherence2) = cm.compare_model_topics(topics)
    self.assertAlmostEqual(np.mean(coherence_topics1), coherence1, 4)
    self.assertAlmostEqual(np.mean(coherence_topics2), coherence2, 4)
    self.assertGreater(coherence1, coherence2)
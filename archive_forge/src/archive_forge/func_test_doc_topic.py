import unittest
import logging
import numpy as np  # for arrays, array broadcasting etc.
from gensim.models import ldaseqmodel
from gensim.corpora import Dictionary
from gensim.test.utils import datapath
def test_doc_topic(self):
    doc_topic = self.ldaseq.doc_topics(0)
    expected_doc_topic = 0.0006657789613848203
    self.assertAlmostEqual(doc_topic[0], expected_doc_topic, places=2)
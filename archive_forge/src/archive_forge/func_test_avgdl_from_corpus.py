from collections import defaultdict
import math
import unittest
from gensim.models.bm25model import BM25ABC
from gensim.models import OkapiBM25Model, LuceneBM25Model, AtireBM25Model
from gensim.corpora import Dictionary
def test_avgdl_from_corpus(self):
    corpus = list(map(self.dictionary.doc2bow, self.documents))
    model = BM25Stub(corpus=corpus)
    actual_avgdl = model.avgdl
    self.assertAlmostEqual(self.expected_avgdl, actual_avgdl)
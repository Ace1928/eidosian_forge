from collections import defaultdict
import math
import unittest
from gensim.models.bm25model import BM25ABC
from gensim.models import OkapiBM25Model, LuceneBM25Model, AtireBM25Model
from gensim.corpora import Dictionary
def get_expected_weight(word):
    idf = model.idfs[self.dictionary.token2id[word]]
    numerator = self.k1 + 1
    denominator = 1 + self.k1 * (1 - self.b + self.b * len(first_document) / model.avgdl)
    return idf * numerator / denominator
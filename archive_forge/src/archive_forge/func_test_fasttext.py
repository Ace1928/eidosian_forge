import logging
import unittest
import math
import os
import numpy
import scipy
from gensim import utils
from gensim.corpora import Dictionary
from gensim.models import word2vec
from gensim.models import doc2vec
from gensim.models import KeyedVectors
from gensim.models import TfidfModel
from gensim import matutils, similarities
from gensim.models import Word2Vec, FastText
from gensim.test.utils import (
from gensim.similarities import UniformTermSimilarityIndex
from gensim.similarities import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.similarities import LevenshteinSimilarityIndex
from gensim.similarities.docsim import _nlargest
from gensim.similarities.fastss import editdist
def test_fasttext(self):

    class LeeReader:

        def __init__(self, fn):
            self.fn = fn

        def __iter__(self):
            with utils.open(self.fn, 'r', encoding='latin_1') as infile:
                for line in infile:
                    yield line.lower().strip().split()
    model = FastText(LeeReader(datapath('lee.cor')), bucket=5000)
    index = self.indexer(model)
    self.assertVectorIsSimilarToItself(model.wv, index)
    self.assertApproxNeighborsMatchExact(model.wv, model.wv, index)
    self.assertIndexSaved(index)
    self.assertLoadedIndexEqual(index, model)
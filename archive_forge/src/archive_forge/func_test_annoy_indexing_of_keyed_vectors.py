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
def test_annoy_indexing_of_keyed_vectors(self):
    from gensim.similarities.annoy import AnnoyIndexer
    keyVectors_file = datapath('lee_fasttext.vec')
    model = KeyedVectors.load_word2vec_format(keyVectors_file)
    index = AnnoyIndexer(model, 10)
    self.assertEqual(index.num_trees, 10)
    self.assertVectorIsSimilarToItself(model, index)
    self.assertApproxNeighborsMatchExact(model, model, index)
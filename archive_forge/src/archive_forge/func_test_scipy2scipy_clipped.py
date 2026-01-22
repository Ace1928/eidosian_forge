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
def test_scipy2scipy_clipped(self):
    vec = [0.8, 0.2, 0.0, 0.0, -0.1, -0.15]
    expected = [(0, 0.8), (1, 0.2), (5, -0.15)]
    vec_scipy = scipy.sparse.csr_matrix(vec)
    vec_scipy_clipped = matutils.scipy2scipy_clipped(vec_scipy, topn=3)
    self.assertTrue(scipy.sparse.issparse(vec_scipy_clipped))
    self.assertTrue(matutils.scipy2sparse(vec_scipy_clipped), expected)
    vec = [0.8, 0.2, 0.0, 0.0, -0.1, -0.15]
    expected = [(0, 0.8), (1, 0.2), (5, -0.15)]
    matrix_scipy = scipy.sparse.csr_matrix([vec] * 3)
    matrix_scipy_clipped = matutils.scipy2scipy_clipped(matrix_scipy, topn=3)
    self.assertTrue(scipy.sparse.issparse(matrix_scipy_clipped))
    self.assertTrue([matutils.scipy2sparse(x) for x in matrix_scipy_clipped], [expected] * 3)
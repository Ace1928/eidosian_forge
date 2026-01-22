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
def test_inner_product_corpus_vector_maintain_true(self):
    """Test the inner product between a corpus and a vector with the ('maintain', True) normalization."""
    expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
    expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
    expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec1, self.vec1))
    expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
    expected_result = numpy.full((3, 1), expected_result)
    result = self.uniform_matrix.inner_product([self.vec1] * 3, self.vec2, normalized=('maintain', True))
    self.assertTrue(isinstance(result, numpy.ndarray))
    self.assertTrue(numpy.allclose(expected_result, result))
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
def test_inner_product_corpus_corpus_default(self):
    """Test the inner product between two corpora with the default normalization."""
    expected_result = 0.0
    expected_result += 2 * 1.0 * 1
    expected_result += 2 * 0.5 * 1
    expected_result += 1 * 0.5 * 1
    expected_result += 1 * 0.5 * 1
    expected_result = numpy.full((3, 2), expected_result)
    result = self.uniform_matrix.inner_product([self.vec1] * 3, [self.vec2] * 2)
    self.assertTrue(isinstance(result, scipy.sparse.csr_matrix))
    self.assertTrue(numpy.allclose(expected_result, result.todense()))
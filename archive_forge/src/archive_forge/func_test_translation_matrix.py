from collections import namedtuple
import unittest
import logging
import numpy as np
import pytest
from scipy.spatial.distance import cosine
from gensim.models.doc2vec import Doc2Vec
from gensim import utils
from gensim.models import translation_matrix
from gensim.models import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
def test_translation_matrix(self):
    model = translation_matrix.BackMappingTranslationMatrix(self.source_doc_vec, self.target_doc_vec, self.train_docs[:5])
    transmat = model.train(self.train_docs[:5])
    self.assertEqual(transmat.shape, (8, 8))
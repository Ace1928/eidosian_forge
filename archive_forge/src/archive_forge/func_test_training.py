from __future__ import with_statement, division
import logging
import unittest
import os
from collections import namedtuple
import numpy as np
from testfixtures import log_capture
from gensim import utils
from gensim.models import doc2vec, keyedvectors
from gensim.test.utils import datapath, get_tmpfile, temporary_file, common_texts as raw_sentences
def test_training(self):
    """Test doc2vec training."""
    corpus = DocsLeeCorpus()
    model = doc2vec.Doc2Vec(vector_size=100, min_count=2, epochs=20, workers=1)
    model.build_vocab(corpus)
    self.assertEqual(model.dv.vectors.shape, (300, 100))
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
    self.model_sanity(model)
    model2 = doc2vec.Doc2Vec(corpus, vector_size=100, min_count=2, epochs=20, workers=1)
    self.models_equal(model, model2)
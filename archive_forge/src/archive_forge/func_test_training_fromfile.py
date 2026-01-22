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
def test_training_fromfile(self):
    """Test doc2vec training."""
    with temporary_file(get_tmpfile('gensim_doc2vec.tst')) as corpus_file:
        save_lee_corpus_as_line_sentence(corpus_file)
        model = doc2vec.Doc2Vec(vector_size=100, min_count=2, epochs=20, workers=1)
        model.build_vocab(corpus_file=corpus_file)
        self.assertEqual(model.dv.vectors.shape, (300, 100))
        model.train(corpus_file=corpus_file, total_words=model.corpus_total_words, epochs=model.epochs)
        self.model_sanity(model)
        model = doc2vec.Doc2Vec(corpus_file=corpus_file, vector_size=100, min_count=2, epochs=20, workers=1)
        self.model_sanity(model)
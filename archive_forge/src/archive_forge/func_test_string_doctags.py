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
def test_string_doctags(self):
    """Test doc2vec doctag alternatives"""
    corpus = list(DocsLeeCorpus(True))
    corpus = corpus[0:10] + corpus
    model = doc2vec.Doc2Vec(min_count=1)
    model.build_vocab(corpus)
    self.assertEqual(len(model.dv.vectors), 300)
    self.assertEqual(model.dv[0].shape, (100,))
    self.assertEqual(model.dv['_*0'].shape, (100,))
    self.assertTrue(all(model.dv['_*0'] == model.dv[0]))
    self.assertTrue(max(model.dv.key_to_index.values()) < len(model.dv.index_to_key))
    self.assertLess(max((model.dv.get_index(str_key) for str_key in model.dv.key_to_index.keys())), len(model.dv.vectors))
    self.assertEqual(model.dv.index_to_key[0], model.dv.most_similar([model.dv[0]])[0][0])
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
@log_capture()
def test_build_vocab_warning(self, loglines):
    """Test if logger warning is raised on non-ideal input to a doc2vec model"""
    raw_sentences = ['human', 'machine']
    sentences = [doc2vec.TaggedDocument(words, [i]) for i, words in enumerate(raw_sentences)]
    model = doc2vec.Doc2Vec()
    model.build_vocab(sentences)
    warning = "Each 'words' should be a list of words (usually unicode strings)."
    self.assertTrue(warning in str(loglines))
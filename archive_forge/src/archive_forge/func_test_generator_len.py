from __future__ import unicode_literals
import json
import logging
import os.path
import unittest
import numpy as np
from gensim import utils
from gensim.scripts.segment_wiki import segment_all_articles, segment_and_write_all_articles
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.word2vec2tensor import word2vec2tensor
from gensim.models import KeyedVectors
def test_generator_len(self):
    expected_num_articles = 106
    num_articles = sum((1 for x in segment_all_articles(self.fname)))
    self.assertEqual(num_articles, expected_num_articles)
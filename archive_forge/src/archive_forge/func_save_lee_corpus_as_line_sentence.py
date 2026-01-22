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
def save_lee_corpus_as_line_sentence(corpus_file):
    utils.save_as_line_sentence((doc.words for doc in DocsLeeCorpus()), corpus_file)
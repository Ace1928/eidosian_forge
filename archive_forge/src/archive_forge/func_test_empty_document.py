import logging
import unittest
import numbers
from os import remove
import numpy as np
from gensim.corpora import mmcorpus, Dictionary
from gensim.models import atmodel
from gensim import matutils
from gensim.test import basetmtests
from gensim.test.utils import (datapath,
from gensim.matutils import jensen_shannon
def test_empty_document(self):
    local_texts = common_texts + [['only_occurs_once_in_corpus_and_alone_in_doc']]
    dictionary = Dictionary(local_texts)
    dictionary.filter_extremes(no_below=2)
    corpus = [dictionary.doc2bow(text) for text in local_texts]
    a2d = author2doc.copy()
    a2d['joaquin'] = [len(local_texts) - 1]
    self.class_(corpus, author2doc=a2d, id2word=dictionary, num_topics=2)
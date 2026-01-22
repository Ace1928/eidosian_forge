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
def test_new_author_topics(self):
    model = self.class_(corpus, author2doc=author2doc, id2word=dictionary, num_topics=2, passes=100, random_state=np.random.seed(0))
    author2doc_newauthor = {}
    author2doc_newauthor['test'] = [0, 1]
    model.update(corpus=corpus[0:2], author2doc=author2doc_newauthor)
    state_gamma_len = len(model.state.gamma)
    author2doc_len = len(model.author2doc)
    author2id_len = len(model.author2id)
    id2author_len = len(model.id2author)
    doc2author_len = len(model.doc2author)
    new_author_topics = model.get_new_author_topics(corpus=corpus[0:2])
    for k, v in new_author_topics:
        self.assertTrue(isinstance(k, int))
        self.assertTrue(isinstance(v, float))
    similarity = 1 / (1 + jensen_shannon(model['test'], new_author_topics))
    self.assertTrue(similarity >= 0.9)
    with self.assertRaises(TypeError):
        model.get_new_author_topics(corpus=corpus[0])
    self.assertEqual(state_gamma_len, len(model.state.gamma))
    self.assertEqual(author2doc_len, len(model.author2doc))
    self.assertEqual(author2id_len, len(model.author2id))
    self.assertEqual(id2author_len, len(model.id2author))
    self.assertEqual(doc2author_len, len(model.doc2author))
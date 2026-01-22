import logging
import numbers
import os
import unittest
import copy
import numpy as np
from numpy.testing import assert_allclose
from gensim.corpora import mmcorpus, Dictionary
from gensim.models import ldamodel, ldamulticore
from gensim import matutils, utils
from gensim.test import basetmtests
from gensim.test.utils import datapath, get_tmpfile, common_texts
@unittest.skipIf(GITHUB_ACTIONS_WINDOWS, 'see <https://github.com/RaRe-Technologies/gensim/pull/2836>')
def test_get_document_topics(self):
    model = self.class_(self.corpus, id2word=dictionary, num_topics=2, passes=100, random_state=np.random.seed(0))
    doc_topics = model.get_document_topics(self.corpus)
    for topic in doc_topics:
        self.assertTrue(isinstance(topic, list))
        for k, v in topic:
            self.assertTrue(isinstance(k, numbers.Integral))
            self.assertTrue(np.issubdtype(v, np.floating))
    all_topics = model.get_document_topics(self.corpus, per_word_topics=True)
    self.assertEqual(model.state.numdocs, len(corpus))
    for topic in all_topics:
        self.assertTrue(isinstance(topic, tuple))
        for k, v in topic[0]:
            self.assertTrue(isinstance(k, numbers.Integral))
            self.assertTrue(np.issubdtype(v, np.floating))
        for w, topic_list in topic[1]:
            self.assertTrue(isinstance(w, numbers.Integral))
            self.assertTrue(isinstance(topic_list, list))
        for w, phi_values in topic[2]:
            self.assertTrue(isinstance(w, numbers.Integral))
            self.assertTrue(isinstance(phi_values, list))
    doc_topic_count_na = 0
    word_phi_count_na = 0
    all_topics = model.get_document_topics(self.corpus, minimum_probability=0.8, minimum_phi_value=1.0, per_word_topics=True)
    self.assertEqual(model.state.numdocs, len(corpus))
    for topic in all_topics:
        self.assertTrue(isinstance(topic, tuple))
        for k, v in topic[0]:
            self.assertTrue(isinstance(k, numbers.Integral))
            self.assertTrue(np.issubdtype(v, np.floating))
            if len(topic[0]) != 0:
                doc_topic_count_na += 1
        for w, topic_list in topic[1]:
            self.assertTrue(isinstance(w, numbers.Integral))
            self.assertTrue(isinstance(topic_list, list))
        for w, phi_values in topic[2]:
            self.assertTrue(isinstance(w, numbers.Integral))
            self.assertTrue(isinstance(phi_values, list))
            if len(phi_values) != 0:
                word_phi_count_na += 1
    self.assertTrue(model.state.numdocs > doc_topic_count_na)
    self.assertTrue(sum((len(i) for i in corpus)) > word_phi_count_na)
    doc_topics, word_topics, word_phis = model.get_document_topics(self.corpus[1], per_word_topics=True)
    for k, v in doc_topics:
        self.assertTrue(isinstance(k, numbers.Integral))
        self.assertTrue(np.issubdtype(v, np.floating))
    for w, topic_list in word_topics:
        self.assertTrue(isinstance(w, numbers.Integral))
        self.assertTrue(isinstance(topic_list, list))
    for w, phi_values in word_phis:
        self.assertTrue(isinstance(w, numbers.Integral))
        self.assertTrue(isinstance(phi_values, list))
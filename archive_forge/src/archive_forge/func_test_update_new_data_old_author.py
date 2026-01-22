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
def test_update_new_data_old_author(self):
    model = self.class_(corpus, author2doc=author2doc, id2word=dictionary, num_topics=2)
    jill_topics = model.get_author_topics('jill')
    jill_topics = matutils.sparse2full(jill_topics, model.num_topics)
    model.update(corpus_new, author2doc_new)
    jill_topics2 = model.get_author_topics('jill')
    jill_topics2 = matutils.sparse2full(jill_topics2, model.num_topics)
    self.assertFalse(all(np.equal(jill_topics, jill_topics2)))
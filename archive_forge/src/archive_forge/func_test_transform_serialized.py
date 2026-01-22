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
def test_transform_serialized(self):
    passed = False
    for i in range(25):
        model = self.class_(id2word=dictionary, num_topics=2, passes=100, random_state=0, serialized=True, serialization_path=datapath('testcorpus_serialization.mm'))
        model.update(self.corpus, author2doc)
        jill_topics = model.get_author_topics('jill')
        vec = matutils.sparse2full(jill_topics, 2)
        expected = [0.91, 0.08]
        passed = np.allclose(sorted(vec), sorted(expected), atol=0.1)
        remove(datapath('testcorpus_serialization.mm'))
        if passed:
            break
        logging.warning('Author-topic model failed to converge on attempt %i (got %s, expected %s)', i, sorted(vec), sorted(expected))
    self.assertTrue(passed)
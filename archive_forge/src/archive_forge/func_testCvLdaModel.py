import logging
import unittest
import multiprocessing as mp
from functools import partial
import numpy as np
from gensim.matutils import argsort
from gensim.models.coherencemodel import CoherenceModel, BOOLEAN_DOCUMENT_BASED
from gensim.models.ldamodel import LdaModel
from gensim.test.utils import get_tmpfile, common_texts, common_dictionary, common_corpus
def testCvLdaModel(self):
    """Perform sanity check to see if c_v coherence works with LDA Model"""
    CoherenceModel(model=self.ldamodel, texts=self.texts, coherence='c_v')
import logging
from itertools import chain
from copy import deepcopy
from shutil import copyfile
from os.path import isfile
from os import remove
import numpy as np  # for arrays, array broadcasting etc.
from scipy.special import gammaln  # gamma function utils
from gensim import utils
from gensim.models import LdaModel
from gensim.models.ldamodel import LdaState
from gensim.matutils import dirichlet_expectation, mean_absolute_difference
from gensim.corpora import MmCorpus
def init_empty_corpus(self):
    """Initialize an empty corpus.
        If the corpora are to be treated as lists, simply initialize an empty list.
        If serialization is used, initialize an empty corpus using :class:`~gensim.corpora.mmcorpus.MmCorpus`.

        """
    if self.serialized:
        MmCorpus.serialize(self.serialization_path, [])
        self.corpus = MmCorpus(self.serialization_path)
    else:
        self.corpus = []
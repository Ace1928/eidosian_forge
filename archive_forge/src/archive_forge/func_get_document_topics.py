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
def get_document_topics(self, word_id, minimum_probability=None):
    """Override :meth:`~gensim.models.ldamodel.LdaModel.get_document_topics` and simply raises an exception.

        Warnings
        --------
        This method invalid for model, use :meth:`~gensim.models.atmodel.AuthorTopicModel.get_author_topics` or
        :meth:`~gensim.models.atmodel.AuthorTopicModel.get_new_author_topics` instead.

        Raises
        ------
        NotImplementedError
            Always.

        """
    raise NotImplementedError('Method "get_document_topics" is not valid for the author-topic model. Use the "get_author_topics" method.')
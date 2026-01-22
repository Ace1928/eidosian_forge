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
def construct_author2doc(doc2author):
    """Make a mapping from author IDs to document IDs.

    Parameters
    ----------
    doc2author: dict of (int, list of str)
        Mapping of document id to authors.

    Returns
    -------
    dict of (str, list of int)
        Mapping of authors to document ids.

    """
    authors_ids = set()
    for d, a_doc_ids in doc2author.items():
        for a in a_doc_ids:
            authors_ids.add(a)
    author2doc = {}
    for a in authors_ids:
        author2doc[a] = []
        for d, a_ids in doc2author.items():
            if a in a_ids:
                author2doc[a].append(d)
    return author2doc
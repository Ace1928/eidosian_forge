import logging
import os
from multiprocessing import Process, Pipe, ProcessError
import importlib
from typing import Set, Optional, List
import numpy as np
from scipy.spatial.distance import cosine
from dataclasses import dataclass
from gensim import utils
from gensim.models import ldamodel, ldamulticore, basemodel
from gensim.utils import SaveLoad
@property
def id2word(self):
    """Return the :py:class:`gensim.corpora.dictionary.Dictionary` object used in the model."""
    return self.gensim_kw_args['id2word']
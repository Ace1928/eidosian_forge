import logging
import sys
import itertools
import warnings
from numbers import Integral
from typing import Iterable
from numpy import (
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.corpora.dictionary import Dictionary
from gensim.utils import deprecated
def resize_vectors(self, seed=0):
    """Make underlying vectors match index_to_key size; random-initialize any new rows."""
    target_shape = (len(self.index_to_key), self.vector_size)
    self.vectors = prep_vectors(target_shape, prior_vectors=self.vectors, seed=seed)
    self.allocate_vecattrs()
    self.norms = None
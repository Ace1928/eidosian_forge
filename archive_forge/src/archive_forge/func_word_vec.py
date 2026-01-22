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
@deprecated('Use get_vector instead')
def word_vec(self, *args, **kwargs):
    """Compatibility alias for get_vector(); must exist so subclass calls reach subclass get_vector()."""
    return self.get_vector(*args, **kwargs)
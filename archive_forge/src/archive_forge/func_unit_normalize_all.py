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
def unit_normalize_all(self):
    """Destructively scale all vectors to unit-length.

        You cannot sensibly continue training after such a step.

        """
    self.fill_norms()
    self.vectors /= self.norms[..., np.newaxis]
    self.norms = np.ones((len(self.vectors),))
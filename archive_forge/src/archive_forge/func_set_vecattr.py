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
def set_vecattr(self, key, attr, val):
    """Set attribute associated with the given key to value.

        Parameters
        ----------

        key : str
            Store the attribute for this vector key.
        attr : str
            Name of the additional attribute to store for the given key.
        val : object
            Value of the additional attribute to store for the given key.

        Returns
        -------

        None

        """
    self.allocate_vecattrs(attrs=[attr], types=[type(val)])
    index = self.get_index(key)
    self.expandos[attr][index] = val
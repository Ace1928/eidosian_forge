import numbers
from functools import reduce
from operator import mul
import numpy as np
@property
def total_nb_rows(self):
    """Total number of rows in this array sequence."""
    return np.sum(self._lengths)
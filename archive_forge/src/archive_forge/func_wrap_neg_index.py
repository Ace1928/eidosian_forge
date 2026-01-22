from __future__ import division
import numbers
from typing import Optional, Tuple
import numpy as np
def wrap_neg_index(index, dim, neg_step: bool=False):
    """Converts a negative index into a positive index.

    Args:
        index: The index to convert. Can be None.
        dim: The length of the dimension being indexed.
    """
    if index is not None and index < 0 and (not (neg_step and index == -1)):
        index += dim
    return index
from __future__ import absolute_import, division
import itertools
import math
import sys
import textwrap
def n_to_indices(num, length):
    """
    Calculate `num` evenly spaced indices for a sequence of length `length`.

    Should be used with num >= 2, result will always include first and last.

    Parameters
    ----------
    num : int
    length : int

    Returns
    -------
    list of int

    """
    if num < 2:
        raise ValueError('num must be 2 or larger, got {0}'.format(num))
    elif num > length:
        raise ValueError('num cannot be greater than length')
    step = (length - 1) / (num - 1)
    return (int(round_(step * i)) for i in range(num))
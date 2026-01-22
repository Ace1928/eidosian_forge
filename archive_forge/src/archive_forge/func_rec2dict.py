from __future__ import annotations
import io
import sys
import typing as ty
import warnings
from functools import reduce
from operator import getitem, mul
from os.path import exists, splitext
import numpy as np
from ._compression import COMPRESSED_FILE_LIKES
from .casting import OK_FLOATS, shared_range
from .externals.oset import OrderedSet
def rec2dict(rec: np.ndarray) -> dict[str, np.generic | np.ndarray]:
    """Convert recarray to dictionary

    Also converts scalar values to scalars

    Parameters
    ----------
    rec : ndarray
       structured ndarray

    Returns
    -------
    dct : dict
       dict with key, value pairs as for `rec`

    Examples
    --------
    >>> r = np.zeros((), dtype = [('x', 'i4'), ('s', 'S10')])
    >>> d = rec2dict(r)
    >>> d == {'x': 0, 's': b''}
    True
    """
    dct = {}
    for key in rec.dtype.fields:
        val = rec[key]
        try:
            val = val.item()
        except ValueError:
            pass
        dct[key] = val
    return dct
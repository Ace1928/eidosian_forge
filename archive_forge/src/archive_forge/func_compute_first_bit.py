from __future__ import annotations
import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object
def compute_first_bit(a):
    """Compute the position of the first nonzero bit for each int in an array."""
    bits = np.bitwise_and.outer(a, 1 << np.arange(32))
    bits = bits.cumsum(axis=1).astype(bool)
    return 33 - bits.sum(axis=1)
import abc
import collections
import io
from typing import (
import numpy as np
import pandas as pd
from cirq import value, ops
from cirq._compat import proper_repr
from cirq.study import resolver
@staticmethod
def dataframe_from_measurements(measurements: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Converts the given measurements to a pandas dataframe.

        This can be used by subclasses as a default implementation for the data
        property. Note that subclasses should typically memoize the result to
        avoid recomputing.
        """
    converted_dict = {}
    for key, bitstrings in measurements.items():
        _, n = bitstrings.shape
        dtype = object if n > 63 else np.int64
        basis = 2 ** np.arange(n, dtype=dtype)[::-1]
        converted_dict[key] = np.sum(basis * bitstrings, axis=1)
    dtype = object if any((bs.shape[1] > 63 for _, bs in measurements.items())) else np.int64
    return pd.DataFrame(converted_dict, dtype=dtype)
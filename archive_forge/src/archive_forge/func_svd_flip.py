from __future__ import annotations
import contextlib
import functools
import itertools
import math
import numbers
import warnings
import numpy as np
from tlz import concat, frequencies
from dask.array.core import Array
from dask.array.numpy_compat import AxisError
from dask.base import is_dask_collection, tokenize
from dask.highlevelgraph import HighLevelGraph
from dask.utils import has_keyword, is_arraylike, is_cupy_type, typename
def svd_flip(u, v, u_based_decision=False):
    """Sign correction to ensure deterministic output from SVD.

    This function is useful for orienting eigenvectors such that
    they all lie in a shared but arbitrary half-space. This makes
    it possible to ensure that results are equivalent across SVD
    implementations and random number generator states.

    Parameters
    ----------

    u : (M, K) array_like
        Left singular vectors (in columns)
    v : (K, N) array_like
        Right singular vectors (in rows)
    u_based_decision: bool
        Whether or not to choose signs based
        on `u` rather than `v`, by default False

    Returns
    -------

    u : (M, K) array_like
        Left singular vectors with corrected sign
    v:  (K, N) array_like
        Right singular vectors with corrected sign
    """
    if u_based_decision:
        dtype = u.dtype
        signs = np.sum(u, axis=0, keepdims=True)
    else:
        dtype = v.dtype
        signs = np.sum(v, axis=1, keepdims=True).T
    signs = 2.0 * ((signs >= 0) - 0.5).astype(dtype)
    u, v = (u * signs, v * signs.T)
    return (u, v)
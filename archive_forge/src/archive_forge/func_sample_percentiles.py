from __future__ import annotations
import math
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_dtype, is_integer_dtype
from tlz import merge, merge_sorted, take
from dask.base import tokenize
from dask.dataframe.core import Series
from dask.dataframe.dispatch import tolist_dispatch
from dask.utils import is_cupy_type, random_state_data
def sample_percentiles(num_old, num_new, chunk_length, upsample=1.0, random_state=None):
    """Construct percentiles for a chunk for repartitioning.

    Adapt the number of total percentiles calculated based on the number
    of current and new partitions.  Returned percentiles include equally
    spaced percentiles between [0, 100], and random percentiles.  See
    detailed discussion below.

    Parameters
    ----------
    num_old: int
        Number of partitions of the current object
    num_new: int
        Number of partitions of the new object
    chunk_length: int
        Number of rows of the partition
    upsample : float
        Multiplicative factor to increase the number of samples

    Returns
    -------
    qs : numpy.ndarray of sorted percentiles between 0, 100

    Constructing ordered (i.e., not hashed) partitions is hard.  Calculating
    approximate percentiles for generic objects in an out-of-core fashion is
    also hard.  Fortunately, partition boundaries don't need to be perfect
    in order for partitioning to be effective, so we strive for a "good enough"
    method that can scale to many partitions and is reasonably well-behaved for
    a wide variety of scenarios.

    Two similar approaches come to mind: (1) take a subsample of every
    partition, then find the best new partitions for the combined subsamples;
    and (2) calculate equally-spaced percentiles on every partition (a
    relatively cheap operation), then merge the results.  We do both, but
    instead of random samples, we use random percentiles.

    If the number of partitions isn't changing, then the ratio of fixed
    percentiles to random percentiles is 2 to 1.  If repartitioning goes from
    a very high number of partitions to a very low number of partitions, then
    we use more random percentiles, because a stochastic approach will be more
    stable to potential correlations in the data that may cause a few equally-
    spaced partitions to under-sample the data.

    The more partitions there are, then the more total percentiles will get
    calculated across all partitions.  Squaring the number of partitions
    approximately doubles the number of total percentiles calculated, so
    num_total_percentiles ~ sqrt(num_partitions).  We assume each partition
    is approximately the same length.  This should provide adequate resolution
    and allow the number of partitions to scale.

    For numeric data, one could instead use T-Digest for floats and Q-Digest
    for ints to calculate approximate percentiles.  Our current method works
    for any dtype.
    """
    random_percentage = 1 / (1 + (4 * num_new / num_old) ** 0.5)
    num_percentiles = upsample * num_new * (num_old + 22) ** 0.55 / num_old
    num_fixed = int(num_percentiles * (1 - random_percentage)) + 2
    num_random = int(num_percentiles * random_percentage) + 2
    if num_fixed + num_random + 5 >= chunk_length:
        return np.linspace(0, 100, chunk_length + 1)
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)
    q_fixed = np.linspace(0, 100, num_fixed)
    q_random = random_state.rand(num_random) * 100
    q_edges = [60 / (num_fixed - 1), 100 - 60 / (num_fixed - 1)]
    qs = np.concatenate([q_fixed, q_random, q_edges, [0, 100]])
    qs.sort()
    qs = 0.5 * (qs[:-1] + qs[1:])
    return qs
import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
def test_all_partitions_concatenated():
    n = np.array([3, 2, 4], dtype=int)
    nc = np.cumsum(n)
    all_partitions = set()
    counter = 0
    for partition_concatenated in _resampling._all_partitions_concatenated(n):
        counter += 1
        partitioning = np.split(partition_concatenated, nc[:-1])
        all_partitions.add(tuple([frozenset(i) for i in partitioning]))
    expected = np.prod([special.binom(sum(n[i:]), sum(n[i + 1:])) for i in range(len(n) - 1)])
    assert_equal(counter, expected)
    assert_equal(len(all_partitions), expected)
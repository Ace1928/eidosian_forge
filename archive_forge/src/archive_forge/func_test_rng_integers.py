from multiprocessing import Pool
from multiprocessing.pool import Pool as PWL
import re
import math
from fractions import Fraction
import numpy as np
from numpy.testing import assert_equal, assert_
import pytest
from pytest import raises as assert_raises
import hypothesis.extra.numpy as npst
from hypothesis import given, strategies, reproduce_failure  # noqa: F401
from scipy.conftest import array_api_compatible
from scipy._lib._array_api import xp_assert_equal
from scipy._lib._util import (_aligned_zeros, check_random_state, MapWrapper,
def test_rng_integers():
    rng = np.random.RandomState()
    arr = rng_integers(rng, low=2, high=5, size=100, endpoint=True)
    assert np.max(arr) == 5
    assert np.min(arr) == 2
    assert arr.shape == (100,)
    arr = rng_integers(rng, low=5, size=100, endpoint=True)
    assert np.max(arr) == 5
    assert np.min(arr) == 0
    assert arr.shape == (100,)
    arr = rng_integers(rng, low=2, high=5, size=100, endpoint=False)
    assert np.max(arr) == 4
    assert np.min(arr) == 2
    assert arr.shape == (100,)
    arr = rng_integers(rng, low=5, size=100, endpoint=False)
    assert np.max(arr) == 4
    assert np.min(arr) == 0
    assert arr.shape == (100,)
    try:
        rng = np.random.default_rng()
    except AttributeError:
        return
    arr = rng_integers(rng, low=2, high=5, size=100, endpoint=True)
    assert np.max(arr) == 5
    assert np.min(arr) == 2
    assert arr.shape == (100,)
    arr = rng_integers(rng, low=5, size=100, endpoint=True)
    assert np.max(arr) == 5
    assert np.min(arr) == 0
    assert arr.shape == (100,)
    arr = rng_integers(rng, low=2, high=5, size=100, endpoint=False)
    assert np.max(arr) == 4
    assert np.min(arr) == 2
    assert arr.shape == (100,)
    arr = rng_integers(rng, low=5, size=100, endpoint=False)
    assert np.max(arr) == 4
    assert np.min(arr) == 0
    assert arr.shape == (100,)
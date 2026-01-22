from __future__ import annotations
import datetime
import functools
import operator
import pickle
from array import array
import pytest
from tlz import curry
from dask import get
from dask.highlevelgraph import HighLevelGraph
from dask.optimization import SubgraphCallable
from dask.utils import (
from dask.utils_test import inc
def test_random_state_data():
    np = pytest.importorskip('numpy')
    seed = 37
    state = np.random.RandomState(seed)
    n = 10000
    states = random_state_data(n, seed)
    assert len(states) == n
    states2 = random_state_data(n, state)
    for s1, s2 in zip(states, states2):
        assert s1.shape == (624,)
        assert (s1 == s2).all()
    states = random_state_data(10, 1234)
    states2 = random_state_data(20, 1234)[:10]
    for s1, s2 in zip(states, states2):
        assert (s1 == s2).all()
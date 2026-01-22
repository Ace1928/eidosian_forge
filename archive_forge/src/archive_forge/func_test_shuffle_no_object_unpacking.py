import warnings
import pytest
import numpy as np
from numpy.testing import (
from numpy import random
import sys
@pytest.mark.parametrize('random', [np.random, np.random.RandomState(), np.random.default_rng()])
@pytest.mark.parametrize('use_array_like', [True, False])
def test_shuffle_no_object_unpacking(self, random, use_array_like):

    class MyArr(np.ndarray):
        pass
    items = [None, np.array([3]), np.float64(3), np.array(10), np.float64(7)]
    arr = np.array(items, dtype=object)
    item_ids = {id(i) for i in items}
    if use_array_like:
        arr = arr.view(MyArr)
    assert all((id(i) in item_ids for i in arr))
    if use_array_like and (not isinstance(random, np.random.Generator)):
        with pytest.warns(UserWarning, match='Shuffling a one dimensional array.*'):
            random.shuffle(arr)
    else:
        random.shuffle(arr)
        assert all((id(i) in item_ids for i in arr))
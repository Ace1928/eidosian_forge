import numpy as np
import pytest
from pandas.core.indexers import (
def test_is_scalar_indexer():
    indexer = (0, 1)
    assert is_scalar_indexer(indexer, 2)
    assert not is_scalar_indexer(indexer[0], 2)
    indexer = (np.array([2]), 1)
    assert not is_scalar_indexer(indexer, 2)
    indexer = (np.array([2]), np.array([3]))
    assert not is_scalar_indexer(indexer, 2)
    indexer = (np.array([2]), np.array([3, 4]))
    assert not is_scalar_indexer(indexer, 2)
    assert not is_scalar_indexer(slice(None), 1)
    indexer = 0
    assert is_scalar_indexer(indexer, 1)
    indexer = (0,)
    assert is_scalar_indexer(indexer, 1)
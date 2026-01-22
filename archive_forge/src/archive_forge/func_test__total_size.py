import pickle
import pytest
import numpy as np
def test__total_size(self):
    """ Test e._total_size """
    e = _ArrayMemoryError((1,), np.dtype(np.uint8))
    assert e._total_size == 1
    e = _ArrayMemoryError((2, 4), np.dtype((np.uint64, 16)))
    assert e._total_size == 1024
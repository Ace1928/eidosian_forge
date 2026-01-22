import random
from copy import copy
import pytest
import networkx as nx
from networkx.utils import (
from networkx.utils.misc import _dict_to_numpy_array1, _dict_to_numpy_array2
def test_create_random_state():
    np = pytest.importorskip('numpy')
    rs = np.random.RandomState
    assert isinstance(create_random_state(1), rs)
    assert isinstance(create_random_state(None), rs)
    assert isinstance(create_random_state(np.random), rs)
    assert isinstance(create_random_state(rs(1)), rs)
    rng = np.random.default_rng()
    assert isinstance(create_random_state(rng), np.random.Generator)
    pytest.raises(ValueError, create_random_state, 'a')
    assert np.all(rs(1).rand(10) == create_random_state(1).rand(10))
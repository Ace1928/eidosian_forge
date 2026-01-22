import random
from copy import copy
import pytest
import networkx as nx
from networkx.utils import (
from networkx.utils.misc import _dict_to_numpy_array1, _dict_to_numpy_array2
def test_numpy_to_list_of_ints(self):
    a = np.array([1, 2, 3], dtype=np.int64)
    b = np.array([1.0, 2, 3])
    c = np.array([1.1, 2, 3])
    assert type(make_list_of_ints(a)) == list
    assert make_list_of_ints(b) == list(b)
    B = make_list_of_ints(b)
    assert type(B[0]) == int
    pytest.raises(nx.NetworkXError, make_list_of_ints, c)
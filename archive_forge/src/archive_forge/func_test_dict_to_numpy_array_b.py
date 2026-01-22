import random
from copy import copy
import pytest
import networkx as nx
from networkx.utils import (
from networkx.utils.misc import _dict_to_numpy_array1, _dict_to_numpy_array2
def test_dict_to_numpy_array_b(self):
    d = {'a': 1, 'b': 2}
    mapping = {'a': 0, 'b': 1}
    a = dict_to_numpy_array(d, mapping=mapping)
    np.testing.assert_allclose(a, np.array([1, 2]))
    a = _dict_to_numpy_array1(d)
    np.testing.assert_allclose(a.sum(), 3)
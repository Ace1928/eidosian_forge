import random
from copy import copy
import pytest
import networkx as nx
from networkx.utils import (
from networkx.utils.misc import _dict_to_numpy_array1, _dict_to_numpy_array2
def test__dict_to_numpy_array1(self):
    d = {'a': 1, 'b': 2}
    a = _dict_to_numpy_array1(d, mapping={'a': 0, 'b': 1})
    np.testing.assert_allclose(a, np.array([1, 2]))
    a = _dict_to_numpy_array1(d, mapping={'b': 0, 'a': 1})
    np.testing.assert_allclose(a, np.array([2, 1]))
    a = _dict_to_numpy_array1(d)
    np.testing.assert_allclose(a.sum(), 3)
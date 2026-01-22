import random
from copy import copy
import pytest
import networkx as nx
from networkx.utils import (
from networkx.utils.misc import _dict_to_numpy_array1, _dict_to_numpy_array2
@pytest.mark.parametrize('iterator', ((i for i in range(3)), iter([1, 2, 3])))
def test_arbitrary_element_raises(iterator):
    """Value error is raised when input is an iterator."""
    with pytest.raises(ValueError, match='from an iterator'):
        arbitrary_element(iterator)
import sys
from copy import deepcopy
from typing import List, Callable, Iterator, Union, Optional, Generic, TypeVar, TYPE_CHECKING
from collections import OrderedDict
def find_data(self, data: str) -> 'Iterator[Tree[_Leaf_T]]':
    """Returns all nodes of the tree whose data equals the given data."""
    return self.find_pred(lambda t: t.data == data)
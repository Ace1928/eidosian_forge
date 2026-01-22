import itertools
from functools import reduce
from operator import mul
from typing import List
import wandb
from wandb import util
from wandb.data_types import Node
def nested_shape(array_or_tuple, seen=None):
    """Figure out the shape of tensors possibly embedded in tuples
    i.e
    [0,0] returns (2)
    ([0,0], [0,0]) returns (2,2)
    (([0,0], [0,0]),[0,0]) returns ((2,2),2)
    """
    if seen is None:
        seen = set()
    if hasattr(array_or_tuple, 'size'):
        return list(array_or_tuple.size())
    elif hasattr(array_or_tuple, 'get_shape'):
        return array_or_tuple.get_shape().as_list()
    elif hasattr(array_or_tuple, 'shape'):
        return array_or_tuple.shape
    seen.add(id(array_or_tuple))
    try:
        return [nested_shape(item, seen) if id(item) not in seen else 0 for item in list(array_or_tuple)]
    except TypeError:
        return []
from typing import Callable, List, Tuple, TypeVar
from ..backends import NumpyOps
from ..config import registry
from ..model import Model
from ..types import Array2d
@registry.layers('list2array.v1')
def list2array() -> Model[InT, OutT]:
    """Transform sequences to ragged arrays if necessary and return the data
    from the ragged array. If sequences are already ragged, do nothing. A
    ragged array is a tuple (data, lengths), where data is the concatenated data.
    """
    return Model('list2array', forward)
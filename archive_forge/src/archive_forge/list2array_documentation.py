from typing import Callable, List, Tuple, TypeVar
from ..backends import NumpyOps
from ..config import registry
from ..model import Model
from ..types import Array2d
Transform sequences to ragged arrays if necessary and return the data
    from the ragged array. If sequences are already ragged, do nothing. A
    ragged array is a tuple (data, lengths), where data is the concatenated data.
    
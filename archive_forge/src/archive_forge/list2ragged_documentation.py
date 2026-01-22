from typing import Callable, List, Tuple, TypeVar, cast
from ..config import registry
from ..model import Model
from ..types import ArrayXd, ListXd, Ragged
Transform sequences to ragged arrays if necessary and return the ragged
    array. If sequences are already ragged, do nothing. A ragged array is a
    tuple (data, lengths), where data is the concatenated data.
    
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union, cast
from ..config import registry
from ..initializers import uniform_init
from ..model import Model
from ..types import Floats1d, Floats2d, Ints1d, Ints2d
from ..util import partial
from .array_getitem import ints_getitem
from .chain import chain

    An embedding layer that uses the “hashing trick” to map keys to distinct values.
    The hashing trick involves hashing each key four times with distinct seeds,
    to produce four likely differing values. Those values are modded into the
    table, and the resulting vectors summed to produce a single result. Because
    it’s unlikely that two different keys will collide on all four “buckets”,
    most distinct keys will receive a distinct vector under this scheme, even
    when the number of vectors in the table is very low.
    
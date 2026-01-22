from typing import Callable, Optional, Tuple, cast
from ..config import registry
from ..initializers import zero_init
from ..model import Model
from ..types import Floats1d, Floats2d
from ..util import ArrayInfo, get_width, partial
def backprop_unnormalized(dY: InT):
    msg = 'backprop is not supported for an unnormalized Softmax layer'
    raise ValueError(msg)
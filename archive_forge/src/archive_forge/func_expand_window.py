from typing import Callable, Tuple, TypeVar, Union, cast
from ..config import registry
from ..model import Model
from ..types import Floats2d, Ragged
@registry.layers('expand_window.v1')
def expand_window(window_size: int=1) -> Model[InT, InT]:
    """For each vector in an input, construct an output vector that contains the
    input and a window of surrounding vectors. This is one step in a convolution.
    """
    return Model('expand_window', forward, attrs={'window_size': window_size})
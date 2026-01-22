from typing import Callable, List, Optional, Tuple, TypeVar, Union, cast
from ..backends import NumpyOps
from ..config import registry
from ..model import Model
from ..types import Array2d, Floats2d, List2d, Padded, Ragged
@registry.layers('with_array2d.v1')
def with_array2d(layer: Model[ValT, ValT], pad: int=0) -> Model[SeqT, SeqT]:
    """Transform sequence data into a contiguous 2d array on the way into and
    out of a model. Handles a variety of sequence types: lists, padded and ragged.
    If the input is a 2d array, it is passed through unchanged.
    """
    return Model(f'with_array({layer.name})', forward, init=init, layers=[layer], attrs={'pad': pad}, dims={name: layer.maybe_get_dim(name) for name in layer.dim_names})
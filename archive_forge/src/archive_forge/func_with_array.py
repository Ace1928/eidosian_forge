from typing import Callable, Optional, Tuple, TypeVar, Union, cast
from ..backends import NumpyOps
from ..config import registry
from ..model import Model
from ..types import Array3d, ArrayXd, ListXd, Padded, Ragged
@registry.layers('with_array.v1')
def with_array(layer: Model[ArrayTXd, ArrayTXd], pad: int=0) -> Model[SeqT, SeqT]:
    """Transform sequence data into a contiguous array on the way into and
    out of a model. Handles a variety of sequence types: lists, padded and ragged.
    If the input is an array, it is passed through unchanged.
    """
    model: Model[SeqT, SeqT] = Model(f'with_array({layer.name})', forward, init=init, layers=[layer], attrs={'pad': pad}, dims={name: layer.maybe_get_dim(name) for name in layer.dim_names})
    return model
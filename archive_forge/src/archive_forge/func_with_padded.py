from typing import Callable, List, Optional, Tuple, TypeVar, Union, cast
from ..config import registry
from ..model import Model
from ..types import Array2d, Floats3d, Ints1d, List2d, Padded, Ragged
from ..util import is_xp_array
@registry.layers('with_padded.v1')
def with_padded(layer: Model[Padded, Padded]) -> Model[SeqT, SeqT]:
    return Model(f'with_padded({layer.name})', forward, init=init, layers=[layer], dims={name: layer.maybe_get_dim(name) for name in layer.dim_names})
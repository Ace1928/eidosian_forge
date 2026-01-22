from typing import Callable, List, Optional, Tuple, TypeVar, Union, cast
from ..backends import NumpyOps
from ..config import registry
from ..model import Model
from ..types import Array2d, Ints1d, List2d, ListXd, Padded, Ragged
@registry.layers('with_ragged.v1')
def with_ragged(layer: Model[Ragged, Ragged]) -> Model[SeqT, SeqT]:
    return Model(f'with_ragged({layer.name})', forward, init=init, layers=[layer])
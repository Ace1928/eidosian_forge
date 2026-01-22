from typing import Callable, Optional, Tuple
import numpy
from ..config import registry
from ..model import Model
from ..types import Floats2d, Ints2d
@registry.layers('uniqued.v1')
def uniqued(layer: Model, *, column: int=0) -> Model[InT, OutT]:
    """Group inputs to a layer, so that the layer only has to compute for the
    unique values. The data is transformed back before output, and the same
    transformation is applied for the gradient. Effectively, this is a cache
    local to each minibatch.
    """
    return Model(f'uniqued({layer.name})', forward, init=init, layers=[layer], dims={'nO': None, 'nI': None}, attrs={'column': column})
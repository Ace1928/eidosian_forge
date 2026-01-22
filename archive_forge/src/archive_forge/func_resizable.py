from typing import Callable, Optional, TypeVar
from ..config import registry
from ..model import Model
from ..types import Floats2d
@registry.layers('resizable.v1')
def resizable(layer, resize_layer: Callable) -> Model[InT, OutT]:
    """Container that holds one layer that can change dimensions."""
    return Model(f'resizable({layer.name})', forward, init=init, layers=[layer], attrs={'resize_layer': resize_layer}, dims={name: layer.maybe_get_dim(name) for name in layer.dim_names})
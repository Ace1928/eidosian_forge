from typing import Any, Callable, Optional, Tuple
from ..config import registry
from ..model import Model
@registry.layers('with_getitem.v1')
def with_getitem(idx: int, layer: Model) -> Model[InT, OutT]:
    """Transform data on the way into and out of a layer, by plucking an item
    from a tuple.
    """
    return Model(f'with_getitem({layer.name})', forward, init=init, layers=[layer], attrs={'idx': idx})
from typing import Any, Optional, Tuple, TypeVar
from ..config import registry
from ..model import Model
def backprop_tuplify(dYs):
    dXs = [bp(dY) for bp, dY in zip(backprops, dYs)]
    dX = dXs[0]
    for dx in dXs[1:]:
        dX += dx
    return dX
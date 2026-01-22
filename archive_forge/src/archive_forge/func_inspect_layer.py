from typing import Dict, Iterable, List, Tuple
from torch import nn
from .namespace import Namespace
def inspect_layer(layer):
    if not isinstance(layer, Skippable):
        return
    for ns, name in layer.stashable():
        stashed_at[ns, name] = j
    for ns, name in layer.poppable():
        prev_j = stashed_at.pop((ns, name))
        skip_routes[ns, name] = (prev_j, j)
from typing import Dict, Iterable, List, Tuple
from torch import nn
from .namespace import Namespace
def inspect_skip_layout(partitions: List[nn.Sequential]) -> SkipLayout:
    """Inspects the skip connection layout in the given partitions."""
    from .skippable import Skippable
    skip_routes: Dict[Tuple[Namespace, str], Tuple[int, int]] = {}
    stashed_at: Dict[Tuple[Namespace, str], int] = {}
    for j, partition in enumerate(partitions):

        def inspect_layer(layer):
            if not isinstance(layer, Skippable):
                return
            for ns, name in layer.stashable():
                stashed_at[ns, name] = j
            for ns, name in layer.poppable():
                prev_j = stashed_at.pop((ns, name))
                skip_routes[ns, name] = (prev_j, j)
        if isinstance(partition, nn.Sequential):
            for layer in partition:
                inspect_layer(layer)
        else:
            inspect_layer(partition)
    return SkipLayout(len(partitions), skip_routes)
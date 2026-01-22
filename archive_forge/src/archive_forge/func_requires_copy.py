from typing import Dict, Iterable, List, Tuple
from torch import nn
from .namespace import Namespace
def requires_copy(self, ns: Namespace, name: str) -> bool:
    """Whether the given namespace and name requires partition-to-partition
        copy or not.
        """
    prev_j, next_j = self.by_ns_name.get((ns, name), (-1, -1))
    return prev_j != next_j
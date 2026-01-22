from dataclasses import dataclass
from typing import Optional, Tuple
import ray
from .common import NodeIdStr
from ray.data._internal.memory_tracing import trace_deallocation
from ray.data.block import Block, BlockMetadata
from ray.data.context import DataContext
from ray.types import ObjectRef
def destroy_if_owned(self) -> int:
    """Clears the object store memory for these blocks if owned.

        Returns:
            The number of bytes freed.
        """
    should_free = self.owns_blocks and DataContext.get_current().eager_free
    for b in self.blocks:
        trace_deallocation(b[0], 'RefBundle.destroy_if_owned', free=should_free)
    return self.size_bytes() if should_free else 0
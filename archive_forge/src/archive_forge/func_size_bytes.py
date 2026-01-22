from dataclasses import dataclass
from typing import Optional, Tuple
import ray
from .common import NodeIdStr
from ray.data._internal.memory_tracing import trace_deallocation
from ray.data.block import Block, BlockMetadata
from ray.data.context import DataContext
from ray.types import ObjectRef
def size_bytes(self) -> int:
    """Size of the blocks of this bundle in bytes."""
    return sum((b[1].size_bytes for b in self.blocks))
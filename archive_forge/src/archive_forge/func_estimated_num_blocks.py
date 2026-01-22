import math
from typing import Iterator, List, Optional, Tuple
import numpy as np
from ray.data._internal.memory_tracing import trace_allocation
from ray.data.block import Block, BlockMetadata
from ray.types import ObjectRef
def estimated_num_blocks(self) -> int:
    """Estimate of `executed_num_blocks()`, without triggering actual execution."""
    return self._estimated_num_blocks or self._num_blocks
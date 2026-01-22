import itertools
from abc import abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, TypeVar, Union
from ray.data._internal.block_batching.block_batching import batch_blocks
from ray.data._internal.execution.interfaces.task_context import TaskContext
from ray.data._internal.output_buffer import BlockOutputBuffer
from ray.data.block import Block, BlockAccessor, DataBatch
def fuse(self, other: 'MapTransformer') -> 'MapTransformer':
    """Fuse two `MapTransformer`s together."""
    assert self._target_max_block_size == other._target_max_block_size or (self._target_max_block_size is None or other._target_max_block_size is None)
    target_max_block_size = self._target_max_block_size or other._target_max_block_size
    self_init_fn = self._init_fn
    other_init_fn = other._init_fn

    def fused_init_fn():
        self_init_fn()
        other_init_fn()
    fused_transform_fns = self._transform_fns + other._transform_fns
    transformer = MapTransformer(fused_transform_fns, init_fn=fused_init_fn)
    transformer.set_target_max_block_size(target_max_block_size)
    return transformer
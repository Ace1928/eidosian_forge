import itertools
from abc import abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, TypeVar, Union
from ray.data._internal.block_batching.block_batching import batch_blocks
from ray.data._internal.execution.interfaces.task_context import TaskContext
from ray.data._internal.output_buffer import BlockOutputBuffer
from ray.data.block import Block, BlockAccessor, DataBatch
def set_transform_fns(self, transform_fns: List[MapTransformFn]) -> None:
    """Set the transform functions."""
    assert len(transform_fns) > 0
    assert transform_fns[0].input_type == MapTransformFnDataType.Block, 'The first transform function must take blocks as input.'
    assert transform_fns[-1].output_type == MapTransformFnDataType.Block, 'The last transform function must output blocks.'
    for i in range(len(transform_fns) - 1):
        assert transform_fns[i].output_type == transform_fns[i + 1].input_type, 'The output type of the previous transform function must match the input type of the next transform function.'
    self._transform_fns = transform_fns
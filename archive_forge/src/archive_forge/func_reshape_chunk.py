import dataclasses
import typing
from typing import Callable, Optional
import cupy
from cupyx.distributed.array import _array
from cupyx.distributed.array import _chunk
from cupyx.distributed.array import _modes
def reshape_chunk(chunk: _chunk._Chunk) -> _chunk._Chunk:
    data = chunk.array.reshape(f_shape(chunk.array.shape))
    index = f_idx(chunk.index)
    updates = [(data, f_idx(idx)) for data, idx in chunk.updates]
    return _chunk._Chunk(data, chunk.ready, index, updates, chunk.prevent_gc)
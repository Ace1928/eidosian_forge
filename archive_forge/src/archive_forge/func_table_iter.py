import copy
import os
from functools import partial
from itertools import groupby
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Tuple, TypeVar, Union
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.types
from . import config
from .utils.logging import get_logger
def table_iter(table: Table, batch_size: int, drop_last_batch=False) -> Iterator[pa.Table]:
    """Iterate over sub-tables of size `batch_size`.

    Args:
        table (`pyarrow.Table`):
            PyArrow table to iterate over.
        batch_size (`int`):
            Size of each sub-table to yield.
        drop_last_batch (`bool`, defaults to `False`):
            Drop the last batch if it is smaller than `batch_size`.
    """
    chunks_buffer = []
    chunks_buffer_size = 0
    for chunk in table.to_reader(max_chunksize=batch_size):
        if len(chunk) == 0:
            continue
        elif chunks_buffer_size + len(chunk) < batch_size:
            chunks_buffer.append(chunk)
            chunks_buffer_size += len(chunk)
            continue
        elif chunks_buffer_size + len(chunk) == batch_size:
            chunks_buffer.append(chunk)
            yield pa.Table.from_batches(chunks_buffer)
            chunks_buffer = []
            chunks_buffer_size = 0
        else:
            cropped_chunk_length = batch_size - chunks_buffer_size
            chunks_buffer.append(chunk.slice(0, cropped_chunk_length))
            yield pa.Table.from_batches(chunks_buffer)
            chunks_buffer = [chunk.slice(cropped_chunk_length, len(chunk) - cropped_chunk_length)]
            chunks_buffer_size = len(chunk) - cropped_chunk_length
    if not drop_last_batch and chunks_buffer:
        yield pa.Table.from_batches(chunks_buffer)
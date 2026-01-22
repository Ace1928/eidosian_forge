import contextlib
import dataclasses
from typing import Any, Iterable, Iterator
from cupy._core.core import ndarray
import cupy._creation.from_data as _creation_from_data
import cupy._creation.basic as _creation_basic
from cupy.cuda.device import Device
from cupy.cuda.stream import Event
from cupy.cuda.stream import Stream
from cupy.cuda.stream import get_current_stream
from cupy.cuda import nccl
from cupyx.distributed._nccl_comm import _get_nccl_dtype_and_count
@contextlib.contextmanager
def on_ready(self) -> Iterator[Stream]:
    with self.array.device:
        stream = get_current_stream()
        stream.wait_event(self.ready)
        yield stream
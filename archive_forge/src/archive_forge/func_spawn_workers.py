from contextlib import contextmanager
from queue import Queue
import sys
from threading import Thread
from types import TracebackType
from typing import TYPE_CHECKING, Callable, Dict, Generator, List, Optional, Tuple, Type, Union, cast
import torch
from .microbatch import Batch
from .stream import AbstractStream, use_device, use_stream
@contextmanager
def spawn_workers(devices: List[torch.device]) -> Generator[Tuple[List[InQueue], List[OutQueue]], None, None]:
    try:
        in_queues, out_queues = create_workers(devices)
        yield (in_queues, out_queues)
    finally:
        pass
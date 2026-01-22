import contextlib
from typing import Sequence
import torch
from torch._custom_op.impl import custom_op
from torch.utils._content_store import ContentStoreReader
@contextlib.contextmanager
def load_tensor_reader(loc):
    global LOAD_TENSOR_READER
    assert LOAD_TENSOR_READER is None
    LOAD_TENSOR_READER = ContentStoreReader(loc, cache=False)
    try:
        yield
    finally:
        LOAD_TENSOR_READER = None
from __future__ import annotations
from pickle import PickleBuffer
from pandas.compat._constants import PY310
def flatten_buffer(b: bytes | bytearray | memoryview | PickleBuffer) -> bytes | bytearray | memoryview:
    """
    Return some 1-D `uint8` typed buffer.

    Coerces anything that does not match that description to one that does
    without copying if possible (otherwise will copy).
    """
    if isinstance(b, (bytes, bytearray)):
        return b
    if not isinstance(b, PickleBuffer):
        b = PickleBuffer(b)
    try:
        return b.raw()
    except BufferError:
        return memoryview(b).tobytes('A')
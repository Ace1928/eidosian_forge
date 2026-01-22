from __future__ import annotations
from typing import Any, Literal, overload
import numpy
def delta_encode(data: bytes | bytearray | numpy.ndarray, /, axis: int=-1, dist: int=1, *, out=None) -> bytes | numpy.ndarray:
    """Encode Delta."""
    if dist != 1:
        raise NotImplementedError(f"delta_encode with dist={dist!r} requires the 'imagecodecs' package")
    if isinstance(data, (bytes, bytearray)):
        data = numpy.frombuffer(data, dtype=numpy.uint8)
        diff = numpy.diff(data, axis=0)
        return numpy.insert(diff, 0, data[0]).tobytes()
    dtype = data.dtype
    if dtype.kind == 'f':
        data = data.view(f'{dtype.byteorder}u{dtype.itemsize}')
    diff = numpy.diff(data, axis=axis)
    key: list[int | slice] = [slice(None)] * data.ndim
    key[axis] = 0
    diff = numpy.insert(diff, 0, data[tuple(key)], axis=axis)
    if not data.dtype.isnative:
        diff = diff.byteswap(True).newbyteorder()
    if dtype.kind == 'f':
        return diff.view(dtype)
    return diff
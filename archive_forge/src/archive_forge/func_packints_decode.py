from __future__ import annotations
from typing import Any, Literal, overload
import numpy
def packints_decode(data: bytes, /, dtype: numpy.dtype | str, bitspersample: int, runlen: int=0, *, out=None) -> numpy.ndarray:
    """Decompress bytes to array of integers.

    This implementation only handles itemsizes 1, 8, 16, 32, and 64 bits.
    Install the Imagecodecs package for decoding other integer sizes.

    Parameters:
        data:
            Data to decompress.
        dtype:
            Numpy boolean or integer type.
        bitspersample:
            Number of bits per integer.
        runlen:
            Number of consecutive integers after which to start at next byte.

    Examples:
        >>> packints_decode(b'a', 'B', 1)
        array([0, 1, 1, 0, 0, 0, 0, 1], dtype=uint8)

    """
    if bitspersample == 1:
        data_array = numpy.frombuffer(data, '|B')
        data_array = numpy.unpackbits(data_array)
        if runlen % 8:
            data_array = data_array.reshape(-1, runlen + (8 - runlen % 8))
            data_array = data_array[:, :runlen].reshape(-1)
        return data_array.astype(dtype)
    if bitspersample in (8, 16, 32, 64):
        return numpy.frombuffer(data, dtype)
    raise NotImplementedError(f"packints_decode of {bitspersample}-bit integers requires the 'imagecodecs' package")
from __future__ import annotations
from typing import Any, Literal, overload
import numpy
def bitorder_decode(data: bytes | bytearray | numpy.ndarray, /, *, out=None, _bitorder: list[Any]=[]) -> bytes | numpy.ndarray:
    """Reverse bits in each byte of bytes or numpy array.

    Decode data where pixels with lower column values are stored in the
    lower-order bits of the bytes (TIFF FillOrder is LSB2MSB).

    Parameters:
        data:
            Data to bit-reversed. If bytes type, a new bit-reversed
            bytes is returned. NumPy arrays are bit-reversed in-place.

    Examples:
        >>> bitorder_decode(b'\\x01\\x64')
        b'\\x80&'
        >>> data = numpy.array([1, 666], dtype='uint16')
        >>> bitorder_decode(data)
        >>> data
        array([  128, 16473], dtype=uint16)

    """
    if not _bitorder:
        _bitorder.append(b'\x00\x80@\xc0 \xa0`\xe0\x10\x90P\xd00\xb0p\xf0\x08\x88H\xc8(\xa8h\xe8\x18\x98X\xd88\xb8x\xf8\x04\x84D\xc4$\xa4d\xe4\x14\x94T\xd44\xb4t\xf4\x0c\x8cL\xcc,\xacl\xec\x1c\x9c\\\xdc<\xbc|\xfc\x02\x82B\xc2"\xa2b\xe2\x12\x92R\xd22\xb2r\xf2\n\x8aJ\xca*\xaaj\xea\x1a\x9aZ\xda:\xbaz\xfa\x06\x86F\xc6&\xa6f\xe6\x16\x96V\xd66\xb6v\xf6\x0e\x8eN\xce.\xaen\xee\x1e\x9e^\xde>\xbe~\xfe\x01\x81A\xc1!\xa1a\xe1\x11\x91Q\xd11\xb1q\xf1\t\x89I\xc9)\xa9i\xe9\x19\x99Y\xd99\xb9y\xf9\x05\x85E\xc5%\xa5e\xe5\x15\x95U\xd55\xb5u\xf5\r\x8dM\xcd-\xadm\xed\x1d\x9d]\xdd=\xbd}\xfd\x03\x83C\xc3#\xa3c\xe3\x13\x93S\xd33\xb3s\xf3\x0b\x8bK\xcb+\xabk\xeb\x1b\x9b[\xdb;\xbb{\xfb\x07\x87G\xc7\'\xa7g\xe7\x17\x97W\xd77\xb7w\xf7\x0f\x8fO\xcf/\xafo\xef\x1f\x9f_\xdf?\xbf\x7f\xff')
        _bitorder.append(numpy.frombuffer(_bitorder[0], dtype=numpy.uint8))
    if isinstance(data, (bytes, bytearray)):
        return data.translate(_bitorder[0])
    try:
        view = data.view('uint8')
        numpy.take(_bitorder[1], view, out=view)
        return data
    except ValueError as exc:
        raise NotImplementedError("bitorder_decode of slices requires the 'imagecodecs' package") from exc
    return None
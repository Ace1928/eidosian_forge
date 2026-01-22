import struct
from passlib import exc
from passlib.utils.compat import join_byte_values, byte_elem_value, \
def expand_des_key(key):
    """convert DES from 7 bytes to 8 bytes (by inserting empty parity bits)"""
    if isinstance(key, bytes):
        if len(key) != 7:
            raise ValueError('key must be 7 bytes in size')
    elif isinstance(key, int_types):
        if key < 0 or key > INT_56_MASK:
            raise ValueError('key must be 56-bit non-negative integer')
        return _unpack64(expand_des_key(_pack56(key)))
    else:
        raise exc.ExpectedTypeError(key, 'bytes or int', 'key')
    key = _unpack56(key)
    return join_byte_values(((key >> shift & 127) << 1 for shift in _EXPAND_ITER))
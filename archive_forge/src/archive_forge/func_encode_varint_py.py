from ._crc32c import crc as crc32c_py
from aiokafka.util import NO_EXTENSIONS
def encode_varint_py(value, write):
    """ Encode an integer to a varint presentation. See
    https://developers.google.com/protocol-buffers/docs/encoding?csw=1#varints
    on how those can be produced.

        Arguments:
            value (int): Value to encode
            write (function): Called per byte that needs to be written

        Returns:
            int: Number of bytes written
    """
    value = value << 1 ^ value >> 63
    if value <= 127:
        write(value)
        return 1
    if value <= 16383:
        write(128 | value & 127)
        write(value >> 7)
        return 2
    if value <= 2097151:
        write(128 | value & 127)
        write(128 | value >> 7 & 127)
        write(value >> 14)
        return 3
    if value <= 268435455:
        write(128 | value & 127)
        write(128 | value >> 7 & 127)
        write(128 | value >> 14 & 127)
        write(value >> 21)
        return 4
    if value <= 34359738367:
        write(128 | value & 127)
        write(128 | value >> 7 & 127)
        write(128 | value >> 14 & 127)
        write(128 | value >> 21 & 127)
        write(value >> 28)
        return 5
    else:
        bits = value & 127
        value >>= 7
        i = 0
        while value:
            write(128 | bits)
            bits = value & 127
            value >>= 7
            i += 1
    write(bits)
    return i
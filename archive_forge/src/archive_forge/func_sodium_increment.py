import nacl.exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def sodium_increment(inp: bytes) -> bytes:
    """
    Increment the value of a byte-sequence interpreted
    as the little-endian representation of a unsigned big integer.

    :param inp: input bytes buffer
    :type inp: bytes
    :return: a byte-sequence representing, as a little-endian
             unsigned big integer, the value ``to_int(inp)``
             incremented by one.
    :rtype: bytes

    """
    ensure(isinstance(inp, bytes), raising=exc.TypeError)
    ln = len(inp)
    buf = ffi.new('unsigned char []', ln)
    ffi.memmove(buf, inp, ln)
    lib.sodium_increment(buf, ln)
    return ffi.buffer(buf, ln)[:]
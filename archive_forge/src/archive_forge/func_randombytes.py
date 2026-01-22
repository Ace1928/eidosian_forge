from nacl import exceptions as exc
from nacl._sodium import ffi, lib
def randombytes(size: int) -> bytes:
    """
    Returns ``size`` number of random bytes from a cryptographically secure
    random source.

    :param size: int
    :rtype: bytes
    """
    buf = ffi.new('unsigned char[]', size)
    lib.randombytes(buf, size)
    return ffi.buffer(buf, size)[:]
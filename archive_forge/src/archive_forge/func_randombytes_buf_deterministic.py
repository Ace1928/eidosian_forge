from nacl import exceptions as exc
from nacl._sodium import ffi, lib
def randombytes_buf_deterministic(size: int, seed: bytes) -> bytes:
    """
    Returns ``size`` number of deterministically generated pseudorandom bytes
    from a seed

    :param size: int
    :param seed: bytes
    :rtype: bytes
    """
    if len(seed) != randombytes_SEEDBYTES:
        raise exc.TypeError('Deterministic random bytes must be generated from 32 bytes')
    buf = ffi.new('unsigned char[]', size)
    lib.randombytes_buf_deterministic(buf, size, seed)
    return ffi.buffer(buf, size)[:]
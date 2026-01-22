import nacl.exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def sodium_unpad(s: bytes, blocksize: int) -> bytes:
    """
    Remove ISO/IEC 7816-4 padding from the input byte array ``s``

    :param s: input bytes string
    :type s: bytes
    :param blocksize:
    :type blocksize: int
    :return: unpadded string
    :rtype: bytes
    """
    ensure(isinstance(s, bytes), raising=exc.TypeError)
    ensure(isinstance(blocksize, int), raising=exc.TypeError)
    s_len = len(s)
    u_len = ffi.new('size_t []', 1)
    rc = lib.sodium_unpad(u_len, s, s_len, blocksize)
    if rc != 0:
        raise exc.CryptoError('Unpadding failure')
    return s[:u_len[0]]
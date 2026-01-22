from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_hash_sha256(message: bytes) -> bytes:
    """
    Hashes and returns the message ``message``.

    :param message: bytes
    :rtype: bytes
    """
    digest = ffi.new('unsigned char[]', crypto_hash_sha256_BYTES)
    rc = lib.crypto_hash_sha256(digest, message, len(message))
    ensure(rc == 0, 'Unexpected library error', raising=exc.RuntimeError)
    return ffi.buffer(digest, crypto_hash_sha256_BYTES)[:]
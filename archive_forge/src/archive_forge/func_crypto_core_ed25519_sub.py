from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_core_ed25519_sub(p: bytes, q: bytes) -> bytes:
    """
    Subtract a point from another on the edwards25519 curve.

    :param p: a :py:data:`.crypto_core_ed25519_BYTES` long bytes sequence
              representing a point on the edwards25519 curve
    :type p: bytes
    :param q: a :py:data:`.crypto_core_ed25519_BYTES` long bytes sequence
              representing a point on the edwards25519 curve
    :type q: bytes
    :return: a point on the edwards25519 curve represented as
             a :py:data:`.crypto_core_ed25519_BYTES` long bytes sequence
    :rtype: bytes
    :raises nacl.exceptions.UnavailableError: If called when using a
        minimal build of libsodium.
    """
    ensure(has_crypto_core_ed25519, 'Not available in minimal build', raising=exc.UnavailableError)
    ensure(isinstance(p, bytes) and isinstance(q, bytes) and (len(p) == crypto_core_ed25519_BYTES) and (len(q) == crypto_core_ed25519_BYTES), 'Each point must be a {} long bytes sequence'.format('crypto_core_ed25519_BYTES'), raising=exc.TypeError)
    r = ffi.new('unsigned char[]', crypto_core_ed25519_BYTES)
    rc = lib.crypto_core_ed25519_sub(r, p, q)
    ensure(rc == 0, 'Unexpected library error', raising=exc.RuntimeError)
    return ffi.buffer(r, crypto_core_ed25519_BYTES)[:]
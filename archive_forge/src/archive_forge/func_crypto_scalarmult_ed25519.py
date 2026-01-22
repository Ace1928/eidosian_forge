from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_scalarmult_ed25519(n: bytes, p: bytes) -> bytes:
    """
    Computes and returns the scalar product of a *clamped* integer ``n``
    and the given group element on the edwards25519 curve.
    The scalar is clamped, as done in the public key generation case,
    by setting to zero the bits in position [0, 1, 2, 255] and setting
    to one the bit in position 254.

    :param n: a :py:data:`.crypto_scalarmult_ed25519_SCALARBYTES` long bytes
              sequence representing a scalar
    :type n: bytes
    :param p: a :py:data:`.crypto_scalarmult_ed25519_BYTES` long bytes sequence
              representing a point on the edwards25519 curve
    :type p: bytes
    :return: a point on the edwards25519 curve, represented as a
             :py:data:`.crypto_scalarmult_ed25519_BYTES` long bytes sequence
    :rtype: bytes
    :raises nacl.exceptions.UnavailableError: If called when using a
        minimal build of libsodium.
    """
    ensure(has_crypto_scalarmult_ed25519, 'Not available in minimal build', raising=exc.UnavailableError)
    ensure(isinstance(n, bytes) and len(n) == crypto_scalarmult_ed25519_SCALARBYTES, 'Input must be a {} long bytes sequence'.format('crypto_scalarmult_ed25519_SCALARBYTES'), raising=exc.TypeError)
    ensure(isinstance(p, bytes) and len(p) == crypto_scalarmult_ed25519_BYTES, 'Input must be a {} long bytes sequence'.format('crypto_scalarmult_ed25519_BYTES'), raising=exc.TypeError)
    q = ffi.new('unsigned char[]', crypto_scalarmult_ed25519_BYTES)
    rc = lib.crypto_scalarmult_ed25519(q, n, p)
    ensure(rc == 0, 'Unexpected library error', raising=exc.RuntimeError)
    return ffi.buffer(q, crypto_scalarmult_ed25519_BYTES)[:]
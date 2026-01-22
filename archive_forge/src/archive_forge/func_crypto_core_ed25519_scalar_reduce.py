from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_core_ed25519_scalar_reduce(s: bytes) -> bytes:
    """
    Reduce integer ``s`` to ``s`` modulo ``L``, where ``L`` is the order
    of the main subgroup.

    :param s: a :py:data:`.crypto_core_ed25519_NONREDUCEDSCALARBYTES`
              long bytes sequence representing an integer
    :type s: bytes
    :return: an integer represented as a
              :py:data:`.crypto_core_ed25519_SCALARBYTES` long bytes sequence
    :rtype: bytes
    :raises nacl.exceptions.UnavailableError: If called when using a
        minimal build of libsodium.
    """
    ensure(has_crypto_core_ed25519, 'Not available in minimal build', raising=exc.UnavailableError)
    ensure(isinstance(s, bytes) and len(s) == crypto_core_ed25519_NONREDUCEDSCALARBYTES, 'Integer s must be a {} long bytes sequence'.format('crypto_core_ed25519_NONREDUCEDSCALARBYTES'), raising=exc.TypeError)
    r = ffi.new('unsigned char[]', crypto_core_ed25519_SCALARBYTES)
    lib.crypto_core_ed25519_scalar_reduce(r, s)
    return ffi.buffer(r, crypto_core_ed25519_SCALARBYTES)[:]
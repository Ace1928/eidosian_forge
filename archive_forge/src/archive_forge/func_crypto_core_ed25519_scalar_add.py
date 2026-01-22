from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_core_ed25519_scalar_add(p: bytes, q: bytes) -> bytes:
    """
    Add integers ``p`` and ``q`` modulo ``L``, where ``L`` is the order of
    the main subgroup.

    :param p: a :py:data:`.crypto_core_ed25519_SCALARBYTES`
              long bytes sequence representing an integer
    :type p: bytes
    :param q: a :py:data:`.crypto_core_ed25519_SCALARBYTES`
              long bytes sequence representing an integer
    :type q: bytes
    :return: an integer represented as a
              :py:data:`.crypto_core_ed25519_SCALARBYTES` long bytes sequence
    :rtype: bytes
    :raises nacl.exceptions.UnavailableError: If called when using a
        minimal build of libsodium.
    """
    ensure(has_crypto_core_ed25519, 'Not available in minimal build', raising=exc.UnavailableError)
    ensure(isinstance(p, bytes) and isinstance(q, bytes) and (len(p) == crypto_core_ed25519_SCALARBYTES) and (len(q) == crypto_core_ed25519_SCALARBYTES), 'Each integer must be a {} long bytes sequence'.format('crypto_core_ed25519_SCALARBYTES'), raising=exc.TypeError)
    r = ffi.new('unsigned char[]', crypto_core_ed25519_SCALARBYTES)
    lib.crypto_core_ed25519_scalar_add(r, p, q)
    return ffi.buffer(r, crypto_core_ed25519_SCALARBYTES)[:]
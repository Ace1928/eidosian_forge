from typing import Tuple
from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_sign_ed25519_pk_to_curve25519(public_key_bytes: bytes) -> bytes:
    """
    Converts a public Ed25519 key (encoded as bytes ``public_key_bytes``) to
    a public Curve25519 key as bytes.

    Raises a ValueError if ``public_key_bytes`` is not of length
    ``crypto_sign_PUBLICKEYBYTES``

    :param public_key_bytes: bytes
    :rtype: bytes
    """
    if len(public_key_bytes) != crypto_sign_PUBLICKEYBYTES:
        raise exc.ValueError('Invalid curve public key')
    curve_public_key_len = crypto_sign_curve25519_BYTES
    curve_public_key = ffi.new('unsigned char[]', curve_public_key_len)
    rc = lib.crypto_sign_ed25519_pk_to_curve25519(curve_public_key, public_key_bytes)
    ensure(rc == 0, 'Unexpected library error', raising=exc.RuntimeError)
    return ffi.buffer(curve_public_key, curve_public_key_len)[:]
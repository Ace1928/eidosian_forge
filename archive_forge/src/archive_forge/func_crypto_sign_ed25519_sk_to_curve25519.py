from typing import Tuple
from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_sign_ed25519_sk_to_curve25519(secret_key_bytes: bytes) -> bytes:
    """
    Converts a secret Ed25519 key (encoded as bytes ``secret_key_bytes``) to
    a secret Curve25519 key as bytes.

    Raises a ValueError if ``secret_key_bytes``is not of length
    ``crypto_sign_SECRETKEYBYTES``

    :param secret_key_bytes: bytes
    :rtype: bytes
    """
    if len(secret_key_bytes) != crypto_sign_SECRETKEYBYTES:
        raise exc.ValueError('Invalid curve secret key')
    curve_secret_key_len = crypto_sign_curve25519_BYTES
    curve_secret_key = ffi.new('unsigned char[]', curve_secret_key_len)
    rc = lib.crypto_sign_ed25519_sk_to_curve25519(curve_secret_key, secret_key_bytes)
    ensure(rc == 0, 'Unexpected library error', raising=exc.RuntimeError)
    return ffi.buffer(curve_secret_key, curve_secret_key_len)[:]
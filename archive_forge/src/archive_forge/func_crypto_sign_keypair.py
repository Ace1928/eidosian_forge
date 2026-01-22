from typing import Tuple
from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_sign_keypair() -> Tuple[bytes, bytes]:
    """
    Returns a randomly generated public key and secret key.

    :rtype: (bytes(public_key), bytes(secret_key))
    """
    pk = ffi.new('unsigned char[]', crypto_sign_PUBLICKEYBYTES)
    sk = ffi.new('unsigned char[]', crypto_sign_SECRETKEYBYTES)
    rc = lib.crypto_sign_keypair(pk, sk)
    ensure(rc == 0, 'Unexpected library error', raising=exc.RuntimeError)
    return (ffi.buffer(pk, crypto_sign_PUBLICKEYBYTES)[:], ffi.buffer(sk, crypto_sign_SECRETKEYBYTES)[:])
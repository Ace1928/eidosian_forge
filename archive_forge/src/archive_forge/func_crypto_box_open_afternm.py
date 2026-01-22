from typing import Tuple
from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_box_open_afternm(ciphertext: bytes, nonce: bytes, k: bytes) -> bytes:
    """
    Decrypts and returns the encrypted message ``ciphertext``, using the shared
    key ``k`` and the nonce ``nonce``.

    :param ciphertext: bytes
    :param nonce: bytes
    :param k: bytes
    :rtype: bytes
    """
    if len(nonce) != crypto_box_NONCEBYTES:
        raise exc.ValueError('Invalid nonce')
    if len(k) != crypto_box_BEFORENMBYTES:
        raise exc.ValueError('Invalid shared key')
    padded = b'\x00' * crypto_box_BOXZEROBYTES + ciphertext
    plaintext = ffi.new('unsigned char[]', len(padded))
    res = lib.crypto_box_open_afternm(plaintext, padded, len(padded), nonce, k)
    ensure(res == 0, 'An error occurred trying to decrypt the message', raising=exc.CryptoError)
    return ffi.buffer(plaintext, len(padded))[crypto_box_ZEROBYTES:]
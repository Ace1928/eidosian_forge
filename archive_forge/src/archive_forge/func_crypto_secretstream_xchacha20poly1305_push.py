from typing import ByteString, Optional, Tuple, cast
from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_secretstream_xchacha20poly1305_push(state: crypto_secretstream_xchacha20poly1305_state, m: bytes, ad: Optional[bytes]=None, tag: int=crypto_secretstream_xchacha20poly1305_TAG_MESSAGE) -> bytes:
    """
    Add an encrypted message to the secret stream.

    :param state: a secretstream state object
    :type state: crypto_secretstream_xchacha20poly1305_state
    :param m: the message to encrypt, the maximum length of an individual
              message is
              :data:`.crypto_secretstream_xchacha20poly1305_MESSAGEBYTES_MAX`.
    :type m: bytes
    :param ad: additional data to include in the authentication tag
    :type ad: bytes or None
    :param tag: the message tag, usually
                :data:`.crypto_secretstream_xchacha20poly1305_TAG_MESSAGE` or
                :data:`.crypto_secretstream_xchacha20poly1305_TAG_FINAL`.
    :type tag: int
    :return: ciphertext
    :rtype: bytes

    """
    ensure(isinstance(state, crypto_secretstream_xchacha20poly1305_state), 'State must be a crypto_secretstream_xchacha20poly1305_state object', raising=exc.TypeError)
    ensure(isinstance(m, bytes), 'Message is not bytes', raising=exc.TypeError)
    ensure(len(m) <= crypto_secretstream_xchacha20poly1305_MESSAGEBYTES_MAX, 'Message is too long', raising=exc.ValueError)
    ensure(ad is None or isinstance(ad, bytes), 'Additional data must be bytes or None', raising=exc.TypeError)
    clen = len(m) + crypto_secretstream_xchacha20poly1305_ABYTES
    if state.rawbuf is None or len(state.rawbuf) < clen:
        state.rawbuf = ffi.new('unsigned char[]', clen)
    if ad is None:
        ad = ffi.NULL
        adlen = 0
    else:
        adlen = len(ad)
    rc = lib.crypto_secretstream_xchacha20poly1305_push(state.statebuf, state.rawbuf, ffi.NULL, m, len(m), ad, adlen, tag)
    ensure(rc == 0, 'Unexpected failure', raising=exc.RuntimeError)
    return ffi.buffer(state.rawbuf, clen)[:]
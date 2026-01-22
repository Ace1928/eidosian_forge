from typing import ByteString, Optional, Tuple, cast
from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_secretstream_xchacha20poly1305_keygen() -> bytes:
    """
    Generate a key for use with
    :func:`.crypto_secretstream_xchacha20poly1305_init_push`.

    """
    keybuf = ffi.new('unsigned char[]', crypto_secretstream_xchacha20poly1305_KEYBYTES)
    lib.crypto_secretstream_xchacha20poly1305_keygen(keybuf)
    return ffi.buffer(keybuf)[:]
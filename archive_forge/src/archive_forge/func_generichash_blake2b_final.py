from typing import NoReturn, TypeVar
from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def generichash_blake2b_final(state: Blake2State) -> bytes:
    """Finalize the blake2b hash state and return the digest.

    :param state: a initialized Blake2bState object as returned from
                     :py:func:`.crypto_generichash_blake2b_init`
    :type state: :py:class:`.Blake2State`
    :return: the blake2 digest of the passed-in data stream
    :rtype: bytes
    """
    ensure(isinstance(state, Blake2State), 'State must be a Blake2State object', raising=exc.TypeError)
    _digest = ffi.new('unsigned char[]', crypto_generichash_BYTES_MAX)
    rc = lib.crypto_generichash_blake2b_final(state._statebuf, _digest, state.digest_size)
    ensure(rc == 0, 'Unexpected failure', raising=exc.RuntimeError)
    return ffi.buffer(_digest, state.digest_size)[:]
from typing import NoReturn, TypeVar
from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def generichash_blake2b_init(key: bytes=b'', salt: bytes=b'', person: bytes=b'', digest_size: int=crypto_generichash_BYTES) -> Blake2State:
    """
    Create a new initialized blake2b hash state

    :param key: must be at most
                :py:data:`.crypto_generichash_KEYBYTES_MAX` long
    :type key: bytes
    :param salt: must be at most
                 :py:data:`.crypto_generichash_SALTBYTES` long;
                 will be zero-padded if needed
    :type salt: bytes
    :param person: must be at most
                   :py:data:`.crypto_generichash_PERSONALBYTES` long:
                   will be zero-padded if needed
    :type person: bytes
    :param digest_size: must be at most
                        :py:data:`.crypto_generichash_BYTES_MAX`;
                        the default digest size is
                        :py:data:`.crypto_generichash_BYTES`
    :type digest_size: int
    :return: a initialized :py:class:`.Blake2State`
    :rtype: object
    """
    _checkparams(digest_size, key, salt, person)
    state = Blake2State(digest_size)
    _salt = ffi.new('unsigned char []', crypto_generichash_SALTBYTES)
    _person = ffi.new('unsigned char []', crypto_generichash_PERSONALBYTES)
    ffi.memmove(_salt, salt, len(salt))
    ffi.memmove(_person, person, len(person))
    rc = lib.crypto_generichash_blake2b_init_salt_personal(state._statebuf, key, len(key), digest_size, _salt, _person)
    ensure(rc == 0, 'Unexpected failure', raising=exc.RuntimeError)
    return state
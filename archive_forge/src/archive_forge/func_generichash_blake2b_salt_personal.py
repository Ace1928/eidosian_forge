from typing import NoReturn, TypeVar
from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def generichash_blake2b_salt_personal(data: bytes, digest_size: int=crypto_generichash_BYTES, key: bytes=b'', salt: bytes=b'', person: bytes=b'') -> bytes:
    """One shot hash interface

    :param data: the input data to the hash function
    :type data: bytes
    :param digest_size: must be at most
                        :py:data:`.crypto_generichash_BYTES_MAX`;
                        the default digest size is
                        :py:data:`.crypto_generichash_BYTES`
    :type digest_size: int
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
    :return: digest_size long digest
    :rtype: bytes
    """
    _checkparams(digest_size, key, salt, person)
    ensure(isinstance(data, bytes), 'Input data must be a bytes sequence', raising=exc.TypeError)
    digest = ffi.new('unsigned char[]', digest_size)
    _salt = ffi.new('unsigned char []', crypto_generichash_SALTBYTES)
    _person = ffi.new('unsigned char []', crypto_generichash_PERSONALBYTES)
    ffi.memmove(_salt, salt, len(salt))
    ffi.memmove(_person, person, len(person))
    rc = lib.crypto_generichash_blake2b_salt_personal(digest, digest_size, data, len(data), key, len(key), _salt, _person)
    ensure(rc == 0, 'Unexpected failure', raising=exc.RuntimeError)
    return ffi.buffer(digest, digest_size)[:]
from binascii import hexlify, unhexlify
from hashlib import sha1
import re
import logging; log = logging.getLogger(__name__)
from passlib.utils import to_unicode, xor_bytes
from passlib.utils.compat import irange, u, \
from passlib.crypto.des import des_encrypt_block
import passlib.utils.handlers as uh
def des_cbc_encrypt(key, value, iv=b'\x00' * 8, pad=b'\x00'):
    """performs des-cbc encryption, returns only last block.

    this performs a specific DES-CBC encryption implementation
    as needed by the Oracle10 hash. it probably won't be useful for
    other purposes as-is.

    input value is null-padded to multiple of 8 bytes.

    :arg key: des key as bytes
    :arg value: value to encrypt, as bytes.
    :param iv: optional IV
    :param pad: optional pad byte

    :returns: last block of DES-CBC encryption of all ``value``'s byte blocks.
    """
    value += pad * (-len(value) % 8)
    hash = iv
    for offset in irange(0, len(value), 8):
        chunk = xor_bytes(hash, value[offset:offset + 8])
        hash = des_encrypt_block(key, chunk)
    return hash
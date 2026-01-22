import re
import struct
from functools import reduce
from Cryptodome.Util.py3compat import (tobytes, bord, _copy_bytes, iter_range,
from Cryptodome.Hash import SHA1, SHA256, HMAC, CMAC, BLAKE2s
from Cryptodome.Util.strxor import strxor
from Cryptodome.Random import get_random_bytes
from Cryptodome.Util.number import size as bit_size, long_to_bytes, bytes_to_long
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib,
def bcrypt_check(password, bcrypt_hash):
    """Verify if the provided password matches the given bcrypt hash.

    Args:
      password (byte string or string):
        The secret password or pass phrase to test.
        It must be at most 72 bytes long.
        It must not contain the zero byte.
        Unicode strings will be encoded as UTF-8.
      bcrypt_hash (byte string, bytearray):
        The reference bcrypt hash the password needs to be checked against.

    Raises:
        ValueError: if the password does not match
    """
    bcrypt_hash = tobytes(bcrypt_hash)
    if len(bcrypt_hash) != 60:
        raise ValueError('Incorrect length of the bcrypt hash: %d bytes instead of 60' % len(bcrypt_hash))
    if bcrypt_hash[:4] != b'$2a$':
        raise ValueError('Unsupported prefix')
    p = re.compile(b'\\$2a\\$([0-9][0-9])\\$([A-Za-z0-9./]{22,22})([A-Za-z0-9./]{31,31})')
    r = p.match(bcrypt_hash)
    if not r:
        raise ValueError('Incorrect bcrypt hash format')
    cost = int(r.group(1))
    if not 4 <= cost <= 31:
        raise ValueError('Incorrect cost')
    salt = _bcrypt_decode(r.group(2))
    bcrypt_hash2 = bcrypt(password, cost, salt)
    secret = get_random_bytes(16)
    mac1 = BLAKE2s.new(digest_bits=160, key=secret, data=bcrypt_hash).digest()
    mac2 = BLAKE2s.new(digest_bits=160, key=secret, data=bcrypt_hash2).digest()
    if mac1 != mac2:
        raise ValueError('Incorrect bcrypt hash')
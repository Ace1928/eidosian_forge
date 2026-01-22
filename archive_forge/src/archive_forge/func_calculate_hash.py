import hashlib
import secrets
import struct
from enum import Enum, IntEnum
from typing import Any, Dict, Optional, Tuple, Union, cast
from pypdf._crypt_providers import (
from ._utils import b_, logger_warning
from .generic import (
@staticmethod
def calculate_hash(R: int, password: bytes, salt: bytes, udata: bytes) -> bytes:
    k = hashlib.sha256(password + salt + udata).digest()
    if R < 6:
        return k
    count = 0
    while True:
        count += 1
        k1 = password + k + udata
        e = aes_cbc_encrypt(k[:16], k[16:32], k1 * 64)
        hash_fn = (hashlib.sha256, hashlib.sha384, hashlib.sha512)[sum(e[:16]) % 3]
        k = hash_fn(e).digest()
        if count >= 64 and e[-1] <= count - 32:
            break
    return k[:32]
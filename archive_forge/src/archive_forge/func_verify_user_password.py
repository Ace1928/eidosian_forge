import hashlib
import secrets
import struct
from enum import Enum, IntEnum
from typing import Any, Dict, Optional, Tuple, Union, cast
from pypdf._crypt_providers import (
from ._utils import b_, logger_warning
from .generic import (
@staticmethod
def verify_user_password(R: int, password: bytes, u_value: bytes, ue_value: bytes) -> bytes:
    """
        See :func:`verify_owner_password`.

        Args:
            R: A number specifying which revision of the standard security
                handler shall be used to interpret this dictionary
            password: The user password
            u_value: A 32-byte string, based on the user password, that shall be
                used in determining whether to prompt the user for a password
                and, if so, whether a valid user or owner password was entered.
            ue_value:

        Returns:
            bytes
        """
    password = password[:127]
    if AlgV5.calculate_hash(R, password, u_value[32:40], b'') != u_value[:32]:
        return b''
    iv = bytes((0 for _ in range(16)))
    tmp_key = AlgV5.calculate_hash(R, password, u_value[40:48], b'')
    return aes_cbc_decrypt(tmp_key, iv, ue_value)
import hashlib
import secrets
import struct
from enum import Enum, IntEnum
from typing import Any, Dict, Optional, Tuple, Union, cast
from pypdf._crypt_providers import (
from ._utils import b_, logger_warning
from .generic import (
def verify_v4(self, password: bytes) -> Tuple[bytes, PasswordType]:
    key = AlgV4.verify_owner_password(password, self.R, self.Length, self.values.O, self.values.U, self.P, self.id1_entry, self.EncryptMetadata)
    if key:
        return (key, PasswordType.OWNER_PASSWORD)
    key = AlgV4.verify_user_password(password, self.R, self.Length, self.values.O, self.values.U, self.P, self.id1_entry, self.EncryptMetadata)
    if key:
        return (key, PasswordType.USER_PASSWORD)
    return (b'', PasswordType.NOT_DECRYPTED)
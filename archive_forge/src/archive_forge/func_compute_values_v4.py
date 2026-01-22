import hashlib
import secrets
import struct
from enum import Enum, IntEnum
from typing import Any, Dict, Optional, Tuple, Union, cast
from pypdf._crypt_providers import (
from ._utils import b_, logger_warning
from .generic import (
def compute_values_v4(self, user_password: bytes, owner_password: bytes) -> None:
    rc4_key = AlgV4.compute_O_value_key(owner_password, self.R, self.Length)
    o_value = AlgV4.compute_O_value(rc4_key, user_password, self.R)
    key = AlgV4.compute_key(user_password, self.R, self.Length, o_value, self.P, self.id1_entry, self.EncryptMetadata)
    u_value = AlgV4.compute_U_value(key, self.R, self.id1_entry)
    self._key = key
    self.values.O = o_value
    self.values.U = u_value
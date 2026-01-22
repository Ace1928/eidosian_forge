import hashlib
import secrets
import struct
from enum import Enum, IntEnum
from typing import Any, Dict, Optional, Tuple, Union, cast
from pypdf._crypt_providers import (
from ._utils import b_, logger_warning
from .generic import (
@staticmethod
def verify_perms(key: bytes, perms: bytes, p: int, metadata_encrypted: bool) -> bool:
    """
        See :func:`verify_owner_password` and :func:`compute_perms_value`.

        Args:
            key: The owner password
            perms:
            p: A set of flags specifying which operations shall be permitted
                when the document is opened with user access.
                If bit 2 is set to 1, all other bits are ignored and all
                operations are permitted.
                If bit 2 is set to 0, permission for operations are based on
                the values of the remaining flags defined in Table 24.
            metadata_encrypted:

        Returns:
            A boolean
        """
    b8 = b'T' if metadata_encrypted else b'F'
    p1 = struct.pack('<I', p) + b'\xff\xff\xff\xff' + b8 + b'adb'
    p2 = aes_ecb_decrypt(key, perms)
    return p1 == p2[:12]
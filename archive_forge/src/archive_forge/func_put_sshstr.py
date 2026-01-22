from __future__ import annotations
import binascii
import enum
import os
import re
import typing
import warnings
from base64 import encodebytes as _base64_encode
from dataclasses import dataclass
from cryptography import utils
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import (
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils
from cryptography.hazmat.primitives.ciphers import (
from cryptography.hazmat.primitives.serialization import (
def put_sshstr(self, val: typing.Union[bytes, _FragList]) -> None:
    """Bytes prefixed with u32 length"""
    if isinstance(val, (bytes, memoryview, bytearray)):
        self.put_u32(len(val))
        self.flist.append(val)
    else:
        self.put_u32(val.size())
        self.flist.extend(val.flist)
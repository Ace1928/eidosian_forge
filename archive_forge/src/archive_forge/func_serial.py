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
def serial(self, serial: int) -> SSHCertificateBuilder:
    if not isinstance(serial, int):
        raise TypeError('serial must be an integer')
    if not 0 <= serial < 2 ** 64:
        raise ValueError('serial must be between 0 and 2**64')
    if self._serial is not None:
        raise ValueError('serial already set')
    return SSHCertificateBuilder(_public_key=self._public_key, _serial=serial, _type=self._type, _key_id=self._key_id, _valid_principals=self._valid_principals, _valid_for_all_principals=self._valid_for_all_principals, _valid_before=self._valid_before, _valid_after=self._valid_after, _critical_options=self._critical_options, _extensions=self._extensions)
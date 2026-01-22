from __future__ import annotations
import binascii
import re
import sys
import typing
import warnings
from cryptography import utils
from cryptography.hazmat.bindings._rust import x509 as rust_x509
from cryptography.x509.oid import NameOID, ObjectIdentifier
@classmethod
def from_rfc4514_string(cls, data: str, attr_name_overrides: typing.Optional[_NameOidMap]=None) -> Name:
    return _RFC4514NameParser(data, attr_name_overrides or {}).parse()
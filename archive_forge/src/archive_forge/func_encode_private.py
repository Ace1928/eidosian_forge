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
def encode_private(self, private_key: ed25519.Ed25519PrivateKey, f_priv: _FragList) -> None:
    """Write Ed25519 private key"""
    public_key = private_key.public_key()
    raw_private_key = private_key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
    raw_public_key = public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)
    f_keypair = _FragList([raw_private_key, raw_public_key])
    self.encode_public(public_key, f_priv)
    f_priv.put_sshstr(f_keypair)
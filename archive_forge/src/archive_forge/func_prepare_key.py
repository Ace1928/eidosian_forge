from __future__ import annotations
import hashlib
import hmac
import json
import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, NoReturn, Union, cast, overload
from .exceptions import InvalidKeyError
from .types import HashlibHash, JWKDict
from .utils import (
def prepare_key(self, key: AllowedOKPKeys | str | bytes) -> AllowedOKPKeys:
    if isinstance(key, (bytes, str)):
        key_str = key.decode('utf-8') if isinstance(key, bytes) else key
        key_bytes = key.encode('utf-8') if isinstance(key, str) else key
        if '-----BEGIN PUBLIC' in key_str:
            key = load_pem_public_key(key_bytes)
        elif '-----BEGIN PRIVATE' in key_str:
            key = load_pem_private_key(key_bytes, password=None)
        elif key_str[0:4] == 'ssh-':
            key = load_ssh_public_key(key_bytes)
    if not isinstance(key, (Ed25519PrivateKey, Ed25519PublicKey, Ed448PrivateKey, Ed448PublicKey)):
        raise InvalidKeyError('Expecting a EllipticCurvePrivateKey/EllipticCurvePublicKey. Wrong key provided for EdDSA algorithms')
    return key
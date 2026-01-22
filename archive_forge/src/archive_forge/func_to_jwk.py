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
@staticmethod
def to_jwk(key: AllowedOKPKeys, as_dict: bool=False) -> Union[JWKDict, str]:
    if isinstance(key, (Ed25519PublicKey, Ed448PublicKey)):
        x = key.public_bytes(encoding=Encoding.Raw, format=PublicFormat.Raw)
        crv = 'Ed25519' if isinstance(key, Ed25519PublicKey) else 'Ed448'
        obj = {'x': base64url_encode(force_bytes(x)).decode(), 'kty': 'OKP', 'crv': crv}
        if as_dict:
            return obj
        else:
            return json.dumps(obj)
    if isinstance(key, (Ed25519PrivateKey, Ed448PrivateKey)):
        d = key.private_bytes(encoding=Encoding.Raw, format=PrivateFormat.Raw, encryption_algorithm=NoEncryption())
        x = key.public_key().public_bytes(encoding=Encoding.Raw, format=PublicFormat.Raw)
        crv = 'Ed25519' if isinstance(key, Ed25519PrivateKey) else 'Ed448'
        obj = {'x': base64url_encode(force_bytes(x)).decode(), 'd': base64url_encode(force_bytes(d)).decode(), 'kty': 'OKP', 'crv': crv}
        if as_dict:
            return obj
        else:
            return json.dumps(obj)
    raise InvalidKeyError('Not a public or private key')
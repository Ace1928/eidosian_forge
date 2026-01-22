from __future__ import annotations
import abc
import typing
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
@property
def digest_size(self) -> int:
    return self._digest_size
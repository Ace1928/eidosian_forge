from __future__ import annotations
import abc
import typing
from cryptography.exceptions import (
from cryptography.hazmat.primitives._cipheralgorithm import CipherAlgorithm
from cryptography.hazmat.primitives.ciphers import modes
def encryptor(self):
    if isinstance(self.mode, modes.ModeWithAuthenticationTag):
        if self.mode.tag is not None:
            raise ValueError('Authentication tag must be None when encrypting.')
    from cryptography.hazmat.backends.openssl.backend import backend
    ctx = backend.create_symmetric_encryption_ctx(self.algorithm, self.mode)
    return self._wrap_ctx(ctx, encrypt=True)
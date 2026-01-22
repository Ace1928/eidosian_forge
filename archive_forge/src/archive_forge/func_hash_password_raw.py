from __future__ import annotations
import os
import warnings
from ._password_hasher import (
from ._typing import Literal
from .low_level import Type, hash_secret, hash_secret_raw, verify_secret
def hash_password_raw(password: bytes, salt: bytes | None=None, time_cost: int=DEFAULT_TIME_COST, memory_cost: int=DEFAULT_MEMORY_COST, parallelism: int=DEFAULT_PARALLELISM, hash_len: int=DEFAULT_HASH_LENGTH, type: Type=Type.I) -> bytes:
    """
    Legacy alias for :func:`argon2.low_level.hash_secret_raw` with default
    parameters.

    .. deprecated:: 16.0.0
        Use :class:`argon2.PasswordHasher` for passwords.
    """
    warnings.warn('argon2.hash_password_raw' + _INSTEAD, DeprecationWarning, stacklevel=2)
    if salt is None:
        salt = os.urandom(DEFAULT_RANDOM_SALT_LENGTH)
    return hash_secret_raw(password, salt, time_cost, memory_cost, parallelism, hash_len, type)
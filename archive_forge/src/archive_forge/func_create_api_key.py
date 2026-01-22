from __future__ import annotations
import os
import time
from pathlib import Path
from functools import lru_cache
from lazyops.utils.logs import logger as _logger, null_logger as _null_logger, Logger
from lazyops.imports._pydantic import BaseSettings
from lazyops.libs import lazyload
from lazyops.libs.proxyobj import ProxyObject
from lazyops.libs.abcs.configs.types import AppEnv
from lazyops.libs.fastapi_utils.types.persistence import TemporaryData
from pydantic import model_validator, computed_field, Field
from ..types.user_roles import UserRole
from ..utils.helpers import get_hashed_key, encrypt_key, decrypt_key, aencrypt_key, adecrypt_key, normalize_audience_name
from typing import List, Optional, Dict, Any, Union, overload, Callable, Tuple, TYPE_CHECKING
def create_api_key(self, user_id: str, prefix: Optional[str]=None, suffix: Optional[str]=None) -> Optional[str]:
    """
        Creates an API Key
        """
    if not self.api_key_enabled:
        raise ValueError('API Keys are not enabled. Please set `api_key_secret_key` and `api_key_access_key`')
    key = encrypt_key(user_id, self.api_key_secret_key, self.api_key_access_key)
    if prefix is None:
        prefix = self.api_key_prefix
    if prefix:
        key = f'{prefix}{key}'
    if suffix is None:
        suffix = self.api_key_suffix
    if suffix:
        key = f'{key}{suffix}'
    return key
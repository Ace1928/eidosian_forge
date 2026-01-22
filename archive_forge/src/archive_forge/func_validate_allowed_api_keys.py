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
def validate_allowed_api_keys(self):
    """
        Validates the allow api keys
        """
    if self.allowed_api_keys is None:
        return
    if isinstance(self.allowed_api_keys, str):
        self.allowed_api_keys = [self.allowed_api_keys]
    if isinstance(self.allowed_api_keys, list):
        allowed_api_keys = {}
        for allowed_api_key in self.allowed_api_keys:
            key, client_name, role = self.parse_allowed_api_key(allowed_api_key)
            allowed_api_keys[key] = {'client_name': client_name, 'role': role}
        self.allowed_api_keys = allowed_api_keys
    elif isinstance(self.allowed_api_keys, dict):
        for key, value in self.allowed_api_keys.items():
            if isinstance(value, str):
                self.allowed_api_keys[key] = {'client_name': 'default', 'role': UserRole.parse_role(value)}
            elif isinstance(value, dict) and 'role' not in value and ('client_name' not in value):
                self.allowed_api_keys[key] = {'client_name': value.get('client_name', 'default'), 'role': UserRole.parse_role(value.get('role', 'API_CLIENT'))}
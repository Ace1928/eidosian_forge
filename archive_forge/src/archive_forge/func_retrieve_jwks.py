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
@lru_cache()
def retrieve_jwks(domain: str) -> Dict[str, Any]:
    """
    Retrieves the JWKs from Auth0
    """
    url = f'https://{domain}/.well-known/jwks.json'
    attempts = 0
    e = None
    while attempts < 3:
        try:
            response = niquests.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            attempts += 1
            _logger.warning(f'Unable to retrieve JWKS from {url}: {e}')
            time.sleep(3 * attempts)
    raise ValueError(f'Unable to retrieve JWKS from {url}')
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
def validate_audiences(self):
    """
        Validates the audience
        """
    if self.audiences:
        if self.enable_multi_audiences is None:
            self.enable_multi_audiences = True
        elif not self.enable_multi_audiences:
            raise ValueError('Multiple Audiences must be enabled to use multiple audiences')
        if self.audience and self.audience not in self.audiences:
            self.audiences.insert(0, self.audience)
    elif not self.audience:
        self.audience = f'https://{self.domain}/userinfo'
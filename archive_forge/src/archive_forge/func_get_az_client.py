from __future__ import annotations
import inspect
from abc import ABC
from urllib.parse import urljoin
from fastapi import HTTPException
from lazyops.libs import lazyload
from lazyops.libs.proxyobj import ProxyObject
from lazyops.utils.helpers import timed_cache
from ..types.errors import InvalidOperationException
from ..types.auth import AuthZeroTokenAuth
from ..types.clients import AuthZeroClientObject
from ..utils.lazy import get_az_settings, logger
from .tokens import ClientCredentialsFlow
from typing import Optional, List, Dict, Any, Union
def get_az_client(self, client_id: str, **kwargs) -> Optional['AuthZeroClientObject']:
    """
        Returns the AuthZero Client
        """
    response = self.hget(f'clients/{client_id}')
    if response.status_code == 200:
        return AuthZeroClientObject.model_validate(response.json())
    logger.warning(f'[{response.status_code}] Error getting client: `{client_id}` {response.text}')
    return None
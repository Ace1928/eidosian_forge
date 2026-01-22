from __future__ import annotations
import asyncio
import random
import inspect
import aiohttpx
import functools
import subprocess
from pydantic import BaseModel
from urllib.parse import urlparse
from lazyops.libs.proxyobj import ProxyObject, proxied
from .base import BaseGlobalClient, cachify
from .utils import aget_root_domain, get_user_agent, http_retry_wrapper
from typing import Optional, Type, TypeVar, Literal, Union, Set, Awaitable, Any, Dict, List, Callable, overload, TYPE_CHECKING
def configure_api_client(self, *args, **kwargs) -> aiohttpx.Client:
    """
        Configures the API Client
        """
    if hasattr(self.settings, 'clients') and hasattr(self.settings.clients, 'http_pool'):
        limits = aiohttpx.Limits(max_connections=self.settings.clients.http_pool.max_connections, max_keepalive_connections=self.settings.clients.http_pool.max_keepalive_connections, keepalive_expiry=self.settings.clients.http_pool.keepalive_expiry)
        timeout = self.http_timeout or self.settings.clients.http_pool.default_timeout
    else:
        limits = aiohttpx.Limits(max_connections=100, max_keepalive_connections=20, keepalive_expiry=60)
        timeout = self.http_timeout or 60
    return aiohttpx.Client(base_url=self.endpoint, limits=limits, timeout=timeout, headers=self.headers, verify=False, **self.api_client_kwargs)
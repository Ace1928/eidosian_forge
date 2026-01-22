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
@property
def pdftotext_enabled(self) -> bool:
    """
        Returns whether pdftotext is enabled
        """
    if self._pdftotext_enabled is None:
        try:
            subprocess.check_output(['which', 'pdftotext'])
            self._pdftotext_enabled = True
        except Exception as e:
            self._pdftotext_enabled = False
    return self._pdftotext_enabled
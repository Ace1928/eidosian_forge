from __future__ import annotations
import functools
from lazyops.libs.pooler import ThreadPooler
from lazyops.libs.logging import logger
from typing import Any, Dict, Optional, TypeVar, Callable, TYPE_CHECKING
def register_posthog_client(client: 'PostHogClient', **kwargs):
    """
    Registers the PostHog Client
    """
    global _ph_client
    if _ph_client is None:
        _ph_client = client
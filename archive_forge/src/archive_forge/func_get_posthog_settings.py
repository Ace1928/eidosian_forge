from __future__ import annotations
import functools
from lazyops.libs.pooler import ThreadPooler
from lazyops.libs.logging import logger
from typing import Any, Dict, Optional, TypeVar, Callable, TYPE_CHECKING
def get_posthog_settings() -> 'PostHogSettings':
    """
    Returns the PostHog Settings
    """
    global _ph_settings
    if _ph_settings is None:
        from .config import PostHogSettings
        _ph_settings = PostHogSettings()
    return _ph_settings
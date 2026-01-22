from __future__ import annotations
from lazyops.libs import lazyload
from lazyops.libs.pooler import ThreadPooler
from lazyops.utils.logs import logger
from lazyops.utils.lazy import lazy_import
from lazyops.utils.helpers import fail_after
from typing import Any, Callable, Dict, List, Optional, Union, Type
def get_az_settings() -> 'AuthZeroSettings':
    """
    Returns the AuthZero Settings
    """
    global _az_settings
    if _az_settings is None:
        from ..configs import AuthZeroSettings
        _az_settings = AuthZeroSettings()
    return _az_settings
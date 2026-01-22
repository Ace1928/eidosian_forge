from __future__ import annotations
import time
import anyio
import inspect
import contextlib 
import functools
import hashlib
from lazyops.types.common import UpperStrEnum
from lazyops.utils import timed_cache
from lazyops.utils.helpers import create_background_task, fail_after
from lazyops.utils.lazy import lazy_import
from lazyops.utils.pooler import ThreadPooler
from lazyops.utils.lazy import get_function_name
from .compat import BaseModel, root_validator, get_pyd_dict
from .base import ENOVAL
from typing import Optional, Dict, Any, Callable, List, Union, TypeVar, Type, overload, TYPE_CHECKING
from aiokeydb.utils.logs import logger
from aiokeydb.utils.helpers import afail_after
def should_invalidate(self, *args, _hits: Optional[int]=None, **kwargs) -> bool:
    """
        Returns whether or not the function should be invalidated
        """
    if self.invalidate_if is not None:
        return self.invalidate_if(*args, **kwargs)
    if self.invalidate_after is not None:
        if _hits and isinstance(self.invalidate_after, int):
            return _hits >= self.invalidate_after
        return self.invalidate_after(*args, _hits=_hits, **kwargs)
    return False
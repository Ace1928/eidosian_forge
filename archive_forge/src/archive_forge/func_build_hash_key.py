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
def build_hash_key(self, *args, **kwargs) -> str:
    """
        Builds the key for the function
        """
    hash_func = self.keybuilder or hash_key
    return hash_func(args=args, kwargs=kwargs, typed=self.typed, exclude_keys=self.exclude_keys, exclude_null_values=self.exclude_null_values_in_hash, exclude_default_values=self.exclude_default_values_in_hash, is_class_method=self.is_class_method)
from __future__ import annotations
import time
import uuid
import typing
import random
import inspect
import functools
import datetime
import itertools
import asyncio
import contextlib
import async_lru
import signal
from pathlib import Path
from frozendict import frozendict
from typing import Dict, Callable, List, Any, Union, Coroutine, TypeVar, Optional, TYPE_CHECKING
from lazyops.utils.logs import default_logger
from lazyops.utils.serialization import (
from lazyops.utils.lazy import (
def build_dict_from_str(data: str, **kwargs) -> Union[List[Any], Dict[str, Any]]:
    """
    Helper to build a dictionary from a string
    """
    import json
    if data.startswith('[') and data.endswith(']') or (data.startswith('{') and data.endswith('}')):
        return json.loads(data)
    return build_dict_from_list(data.split(','), **kwargs)
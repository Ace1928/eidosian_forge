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
def build_dict_from_list(data: List[str], seperator: str='=') -> Dict[str, Any]:
    """
    Builds a dictionary from a list of strings
    """
    import json
    return json.loads(str(dict([item.split(seperator) for item in data])))
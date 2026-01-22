from __future__ import annotations
import abc
import time
import asyncio
import functools
from lazyops.imports._niquests import resolve_niquests
import niquests
from lazyops.libs.pooler import ThreadPooler
from lazyops.utils.logs import logger, null_logger, Logger
from lazyops.utils.times import Timer
from typing import Optional, Dict, Any, List, Union, Type, Set, Tuple, Callable, TypeVar, TYPE_CHECKING
from .config import PostHogSettings
from .utils import get_posthog_settings, register_posthog_client, get_posthog_client, has_existing_posthog_client
from .types import PostHogAuth, PostHogEndpoint, EventQueue, EventT
@property
def task_queue(self) -> asyncio.Queue:
    """
        Returns the Task Queue
        """
    if self._task_queue is None:
        self._task_queue = asyncio.Queue()
    return self._task_queue
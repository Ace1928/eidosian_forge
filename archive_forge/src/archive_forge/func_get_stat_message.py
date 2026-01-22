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
def get_stat_message(self) -> str:
    """
        Returns the stat message
        """
    m = f'Total Uptime: |g|{self.started_time.total_s}|e|. Events Sent: |g|{self.events_sent}|e|. Total Request Duration: |g|{self.started_time.pformat_duration(self.send_duration, short=2, include_ms=True)}|e|.'
    if self.send_duration:
        m += f' Avg |g|{self.events_sent / self.send_duration:.2f} events/s|e|'
    return m
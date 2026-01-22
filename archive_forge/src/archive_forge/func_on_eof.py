import asyncio
import collections
import warnings
from typing import (
from .base_protocol import BaseProtocol
from .helpers import BaseTimerContext, TimerNoop, set_exception, set_result
from .log import internal_logger
def on_eof(self, callback: Callable[[], None]) -> None:
    try:
        callback()
    except Exception:
        internal_logger.exception('Exception in eof callback')
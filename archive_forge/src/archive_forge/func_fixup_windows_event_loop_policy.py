from __future__ import annotations
import logging # isort:skip
import sys
import threading
from collections import defaultdict
from traceback import format_exception
from typing import (
import tornado
from tornado import gen
from ..core.types import ID
def fixup_windows_event_loop_policy() -> None:
    if sys.platform == 'win32' and sys.version_info[:3] >= (3, 8, 0) and (tornado.version_info < (6, 1)):
        import asyncio
        if type(asyncio.get_event_loop_policy()) is asyncio.WindowsProactorEventLoopPolicy:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
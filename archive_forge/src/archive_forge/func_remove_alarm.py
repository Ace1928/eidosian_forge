from __future__ import annotations
import asyncio
import functools
import logging
import sys
import typing
from .abstract_loop import EventLoop, ExitMainLoop
def remove_alarm(self, handle) -> bool:
    """
        Remove an alarm.

        Returns True if the alarm exists, False otherwise
        """
    existed = not handle.cancelled()
    handle.cancel()
    return existed
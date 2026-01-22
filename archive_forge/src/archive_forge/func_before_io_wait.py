from __future__ import annotations
import logging
import typing
import exceptiongroup
import trio
from .abstract_loop import EventLoop, ExitMainLoop
def before_io_wait(self, timeout: float) -> None:
    if timeout > 0:
        for idle_callback in self.idle_callbacks.values():
            idle_callback()
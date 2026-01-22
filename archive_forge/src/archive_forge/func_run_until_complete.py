from PySide6.QtCore import (QCoreApplication, QDateTime, QDeadlineTimer,
from . import futures
from . import tasks
import asyncio
import collections.abc
import concurrent.futures
import contextvars
import enum
import os
import signal
import socket
import subprocess
import typing
import warnings
def run_until_complete(self, future: futures.QAsyncioFuture) -> typing.Any:
    if self.is_closed():
        raise RuntimeError('Event loop is closed')
    if self.is_running():
        raise RuntimeError('Event loop is already running')
    arg_was_coro = not asyncio.futures.isfuture(future)
    future = asyncio.tasks.ensure_future(future, loop=self)
    future.add_done_callback(self._run_until_complete_cb)
    self._future_to_complete = future
    try:
        self.run_forever()
    except Exception as e:
        if arg_was_coro and future.done() and (not future.cancelled()):
            future.exception()
        raise e
    finally:
        future.remove_done_callback(self._run_until_complete_cb)
    if not future.done():
        raise RuntimeError('Event loop stopped before Future completed')
    return future.result()
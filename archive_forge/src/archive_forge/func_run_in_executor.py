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
def run_in_executor(self, executor: typing.Optional[concurrent.futures.ThreadPoolExecutor], func: typing.Callable, *args: typing.Tuple) -> asyncio.futures.Future:
    if self.is_closed():
        raise RuntimeError('Event loop is closed')
    if executor is None:
        executor = self._default_executor
    wrapper = QAsyncioExecutorWrapper(func, *args)
    return asyncio.futures.wrap_future(executor.submit(wrapper.do), loop=self)
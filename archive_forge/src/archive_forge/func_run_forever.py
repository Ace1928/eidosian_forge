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
def run_forever(self) -> None:
    if self.is_closed():
        raise RuntimeError('Event loop is closed')
    if self.is_running():
        raise RuntimeError('Event loop is already running')
    asyncio.events._set_running_loop(self)
    self._application.exec()
    asyncio.events._set_running_loop(None)
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
def set_task_factory(self, factory: typing.Optional[typing.Callable]) -> None:
    if factory is not None and (not callable(factory)):
        raise TypeError('The task factory must be a callable or None')
    self._task_factory = factory
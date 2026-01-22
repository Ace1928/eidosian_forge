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
def call_soon(self, callback: typing.Callable, *args: typing.Any, context: typing.Optional[contextvars.Context]=None) -> asyncio.Handle:
    return self._call_soon_impl(callback, *args, context=context, is_threadsafe=False)
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
def get_child_watcher(self) -> asyncio.AbstractChildWatcher:
    raise DeprecationWarning('Child watchers are deprecated since Python 3.12')
import asyncio
import os
import multiprocessing
import signal
import subprocess
import sys
import time
from binascii import hexlify
from tornado.concurrent import (
from tornado import ioloop
from tornado.iostream import PipeIOStream
from tornado.log import gen_log
import typing
from typing import Optional, Any, Callable
@classmethod
def uninitialize(cls) -> None:
    """Removes the ``SIGCHLD`` handler."""
    if not cls._initialized:
        return
    loop = asyncio.get_event_loop()
    loop.remove_signal_handler(signal.SIGCHLD)
    cls._initialized = False
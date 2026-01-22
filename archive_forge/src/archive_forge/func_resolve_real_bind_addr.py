import os
import io
import re
import email.utils
import socket
import sys
import time
import traceback as traceback_
import logging
import platform
import queue
import contextlib
import threading
import urllib.parse
from functools import lru_cache
from . import connections, errors, __version__
from ._compat import bton
from ._compat import IS_PPC
from .workers import threadpool
from .makefile import MakeFile, StreamWriter
@staticmethod
def resolve_real_bind_addr(socket_):
    """Retrieve actual bind address from bound socket."""
    bind_addr = socket_.getsockname()
    if socket_.family in (socket.AF_INET, socket.AF_INET6):
        "UNIX domain sockets are strings or bytes.\n\n            In case of bytes with a leading null-byte it's an abstract socket.\n            "
        return bind_addr[:2]
    if isinstance(bind_addr, bytes):
        bind_addr = bton(bind_addr)
    return bind_addr
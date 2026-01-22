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
def safe_start(self):
    """Run the server forever, and stop it cleanly on exit."""
    try:
        self.start()
    except (KeyboardInterrupt, IOError):
        self.error_log('Keyboard Interrupt: shutting down')
        self.stop()
        raise
    except SystemExit:
        self.error_log('SystemExit raised: shutting down')
        self.stop()
        raise
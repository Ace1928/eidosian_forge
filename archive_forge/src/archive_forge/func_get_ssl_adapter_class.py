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
def get_ssl_adapter_class(name='builtin'):
    """Return an SSL adapter class for the given name."""
    adapter = ssl_adapters[name.lower()]
    if isinstance(adapter, str):
        last_dot = adapter.rfind('.')
        attr_name = adapter[last_dot + 1:]
        mod_path = adapter[:last_dot]
        try:
            mod = sys.modules[mod_path]
            if mod is None:
                raise KeyError()
        except KeyError:
            mod = __import__(mod_path, globals(), locals(), [''])
        try:
            adapter = getattr(mod, attr_name)
        except AttributeError:
            raise AttributeError("'%s' object has no attribute '%s'" % (mod_path, attr_name))
    return adapter
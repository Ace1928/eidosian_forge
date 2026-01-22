import atexit
import contextlib
import functools
import inspect
import io
import os
import platform
import sys
import threading
import traceback
import debugpy
from debugpy.common import json, timestamp, util
def reraise_exception(format_string='', *args, **kwargs):
    """Like swallow_exception(), but re-raises the current exception after logging it."""
    assert 'exc_info' not in kwargs
    _exception(format_string, *args, **kwargs)
    raise
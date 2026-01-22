import atexit
import os
import platform
import random
import sys
import threading
import time
import uuid
from collections import deque
import sentry_sdk
from sentry_sdk._compat import PY33, PY311
from sentry_sdk._lru_cache import LRUCache
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import (
def extract_frame(fid, raw_frame, cwd):
    abs_path = raw_frame.f_code.co_filename
    try:
        module = raw_frame.f_globals['__name__']
    except Exception:
        module = None
    return {'abs_path': os.path.join(cwd, abs_path), 'module': module, 'filename': filename_for_module(module, abs_path) or None, 'function': fid[2], 'lineno': raw_frame.f_lineno}
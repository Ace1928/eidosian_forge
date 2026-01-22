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
def get_frame_name(frame):
    f_code = frame.f_code
    co_varnames = f_code.co_varnames
    name = f_code.co_name
    try:
        if co_varnames and co_varnames[0] == 'self' and ('self' in frame.f_locals):
            for cls in frame.f_locals['self'].__class__.__mro__:
                if name in cls.__dict__:
                    return '{}.{}'.format(cls.__name__, name)
    except AttributeError:
        pass
    try:
        if co_varnames and co_varnames[0] == 'cls' and ('cls' in frame.f_locals):
            for cls in frame.f_locals['cls'].__mro__:
                if name in cls.__dict__:
                    return '{}.{}'.format(cls.__name__, name)
    except AttributeError:
        pass
    return name
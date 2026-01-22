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
def frame_id(raw_frame):
    return (raw_frame.f_code.co_filename, raw_frame.f_lineno, get_frame_name(raw_frame))
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
def update_active_thread_id(self):
    self.active_thread_id = get_current_thread_id()
    logger.debug('[Profiling] updating active thread id to {tid}'.format(tid=self.active_thread_id))
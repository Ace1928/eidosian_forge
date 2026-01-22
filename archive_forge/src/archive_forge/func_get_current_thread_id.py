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
def get_current_thread_id(thread=None):
    """
    Try to get the id of the current thread, with various fall backs.
    """
    if thread is not None:
        try:
            thread_id = thread.ident
            if thread_id is not None:
                return thread_id
        except AttributeError:
            pass
    if is_gevent():
        gevent_hub = get_gevent_hub()
        if gevent_hub is not None:
            try:
                return gevent_hub.thread_ident
            except AttributeError:
                pass
    try:
        current_thread_id = threading.current_thread().ident
        if current_thread_id is not None:
            return current_thread_id
    except AttributeError:
        pass
    try:
        main_thread_id = threading.main_thread().ident
        if main_thread_id is not None:
            return main_thread_id
    except AttributeError:
        pass
    return None
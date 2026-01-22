from _pydev_bundle._pydev_saved_modules import threading
from _pydev_bundle import _pydev_saved_modules
from _pydevd_bundle.pydevd_utils import notify_about_gevent_if_needed
import weakref
from _pydevd_bundle.pydevd_constants import IS_JYTHON, IS_IRONPYTHON, \
from _pydev_bundle.pydev_log import exception as pydev_log_exception
import sys
from _pydev_bundle import pydev_log
import pydevd_tracing
from _pydevd_bundle.pydevd_collect_bytecode_info import iter_instructions
def new_pick_some_non_daemon_thread():
    with _active_limbo_lock:
        threads = list(_active.values()) + list(_limbo.values())
    for t in threads:
        if not t.daemon and t.is_alive():
            return t
    return None
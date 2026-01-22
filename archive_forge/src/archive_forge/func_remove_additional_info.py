from _pydevd_bundle.pydevd_constants import (STATE_RUN, PYTHON_SUSPEND, SUPPORT_GEVENT, ForkSafeLock,
from _pydev_bundle import pydev_log
from _pydev_bundle._pydev_saved_modules import threading
import weakref
def remove_additional_info(info):
    with _update_infos_lock:
        _all_infos.discard(info)
        _infos_stepping.discard(info)
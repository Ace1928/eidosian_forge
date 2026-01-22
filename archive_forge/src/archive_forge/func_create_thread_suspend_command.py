from contextlib import contextmanager
import sys
from _pydevd_bundle.pydevd_constants import get_frame, RETURN_VALUES_DICT, \
from _pydevd_bundle.pydevd_xml import get_variable_details, get_type
from _pydev_bundle.pydev_override import overrides
from _pydevd_bundle.pydevd_resolver import sorted_attributes_key, TOO_LARGE_ATTR, get_var_scope
from _pydevd_bundle.pydevd_safe_repr import SafeRepr
from _pydev_bundle import pydev_log
from _pydevd_bundle import pydevd_vars
from _pydev_bundle.pydev_imports import Exec
from _pydevd_bundle.pydevd_frame_utils import FramesList
from _pydevd_bundle.pydevd_utils import ScopeRequest, DAPGrouper, Timer
from typing import Optional
def create_thread_suspend_command(self, thread_id, stop_reason, message, suspend_type):
    with self._lock:
        frames_list = self._thread_id_to_frames_list[thread_id]
        cmd = self.py_db.cmd_factory.make_thread_suspend_message(self.py_db, thread_id, frames_list, stop_reason, message, suspend_type)
        frames_list = None
        return cmd
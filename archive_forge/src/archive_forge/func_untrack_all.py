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
def untrack_all(self):
    with self._lock:
        if self._untracked:
            return
        self._untracked = True
        for thread_id in self._thread_id_to_frame_ids:
            self._suspended_frames_manager._thread_id_to_tracker.pop(thread_id, None)
        for frame_id in self._frame_id_to_frame:
            del self._suspended_frames_manager._variable_reference_to_frames_tracker[frame_id]
        self._frame_id_to_frame.clear()
        self._frame_id_to_main_thread_id.clear()
        self._thread_id_to_frame_ids.clear()
        self._thread_id_to_frames_list.clear()
        self._main_thread_id = None
        self._suspended_frames_manager = None
        self._variable_reference_to_variable.clear()
from _pydev_bundle.pydev_is_thread_alive import is_thread_alive
from _pydev_bundle.pydev_log import exception as pydev_log_exception
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle.pydevd_constants import (get_current_thread_id, NO_FTRACE,
from pydevd_file_utils import get_abs_path_real_path_and_base_from_frame, NORM_PATHS_AND_BASE_CONTAINER
from _pydevd_bundle.pydevd_frame import PyDBFrame, is_unhandled_exception
def trace_dispatch_and_unhandled_exceptions(self, frame, event, arg):
    frame_trace_dispatch = self._frame_trace_dispatch
    if frame_trace_dispatch is not None:
        self._frame_trace_dispatch = frame_trace_dispatch(frame, event, arg)
    if event == 'exception':
        self._last_exc_arg = arg
        self._raise_lines.add(frame.f_lineno)
        self._last_raise_line = frame.f_lineno
    elif event == 'return' and self._last_exc_arg is not None:
        try:
            py_db, t, additional_info = self._args[0:3]
            if not additional_info.suspended_at_unhandled:
                if is_unhandled_exception(self, py_db, frame, self._last_raise_line, self._raise_lines):
                    py_db.stop_on_unhandled_exception(py_db, t, additional_info, self._last_exc_arg)
        finally:
            self._last_exc_arg = None
    ret = self.trace_dispatch_and_unhandled_exceptions
    frame.f_trace = ret
    return ret
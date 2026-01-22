from _pydevd_bundle import pydevd_utils
from _pydevd_bundle.pydevd_additional_thread_info import set_additional_thread_info
from _pydevd_bundle.pydevd_comm_constants import CMD_STEP_INTO, CMD_THREAD_SUSPEND
from _pydevd_bundle.pydevd_constants import PYTHON_SUSPEND, STATE_SUSPEND, get_thread_id, STATE_RUN
from _pydev_bundle._pydev_saved_modules import threading
from _pydev_bundle import pydev_log
def suspend_all_threads(py_db, except_thread):
    """
    Suspend all except the one passed as a parameter.
    :param except_thread:
    """
    pydev_log.info('Suspending all threads except: %s', except_thread)
    all_threads = pydevd_utils.get_non_pydevd_threads()
    for t in all_threads:
        if getattr(t, 'pydev_do_not_trace', None):
            pass
        else:
            if t is except_thread:
                continue
            info = mark_thread_suspended(t, CMD_THREAD_SUSPEND)
            frame = info.get_topmost_frame(t)
            if frame is not None:
                try:
                    py_db.set_trace_for_frame_and_parents(frame)
                finally:
                    frame = None
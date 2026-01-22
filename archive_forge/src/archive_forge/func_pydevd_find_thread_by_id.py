from _pydevd_bundle import pydevd_utils
from _pydevd_bundle.pydevd_additional_thread_info import set_additional_thread_info
from _pydevd_bundle.pydevd_comm_constants import CMD_STEP_INTO, CMD_THREAD_SUSPEND
from _pydevd_bundle.pydevd_constants import PYTHON_SUSPEND, STATE_SUSPEND, get_thread_id, STATE_RUN
from _pydev_bundle._pydev_saved_modules import threading
from _pydev_bundle import pydev_log
def pydevd_find_thread_by_id(thread_id):
    try:
        threads = threading.enumerate()
        for i in threads:
            tid = get_thread_id(i)
            if thread_id == tid or thread_id.endswith('|' + tid):
                return i
        pydev_log.info('Could not find thread %s.', thread_id)
        pydev_log.info('Available: %s.', ([get_thread_id(t) for t in threads],))
    except:
        pydev_log.exception()
    return None
from _pydevd_bundle import pydevd_utils
from _pydevd_bundle.pydevd_additional_thread_info import set_additional_thread_info
from _pydevd_bundle.pydevd_comm_constants import CMD_STEP_INTO, CMD_THREAD_SUSPEND
from _pydevd_bundle.pydevd_constants import PYTHON_SUSPEND, STATE_SUSPEND, get_thread_id, STATE_RUN
from _pydev_bundle._pydev_saved_modules import threading
from _pydev_bundle import pydev_log
def internal_run_thread(thread, set_additional_thread_info):
    info = set_additional_thread_info(thread)
    info.pydev_original_step_cmd = -1
    info.pydev_step_cmd = -1
    info.pydev_step_stop = None
    info.pydev_state = STATE_RUN
import json
from _pydev_bundle.pydev_is_thread_alive import is_thread_alive
from _pydev_bundle._pydev_saved_modules import thread
from _pydevd_bundle import pydevd_xml, pydevd_frame_utils, pydevd_constants, pydevd_utils
from _pydevd_bundle.pydevd_comm_constants import (
from _pydevd_bundle.pydevd_constants import (DebugInfoHolder, get_thread_id,
from _pydevd_bundle.pydevd_net_command import NetCommand, NULL_NET_COMMAND, NULL_EXIT_COMMAND
from _pydevd_bundle.pydevd_utils import quote_smart as quote, get_non_pydevd_threads
from pydevd_file_utils import get_abs_path_real_path_and_base_from_frame
import pydevd_file_utils
from pydevd_tracing import get_exception_traceback_str
from _pydev_bundle._pydev_completer import completions_to_xml
from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_frame_utils import FramesList
from io import StringIO
def make_show_console_message(self, py_db, thread_id, frame):
    try:
        frames_list = pydevd_frame_utils.create_frames_list_from_frame(frame)
        thread_suspended_str, _thread_stack_str = self.make_thread_suspend_str(py_db, thread_id, frames_list, CMD_SHOW_CONSOLE, '')
        return NetCommand(CMD_SHOW_CONSOLE, 0, thread_suspended_str)
    except:
        return self.make_error_message(0, get_exception_traceback_str())
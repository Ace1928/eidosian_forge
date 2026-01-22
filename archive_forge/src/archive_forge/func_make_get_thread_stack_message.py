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
def make_get_thread_stack_message(self, py_db, seq, thread_id, topmost_frame, fmt, must_be_suspended=False, start_frame=0, levels=0):
    """
        Returns thread stack as XML.

        :param must_be_suspended: If True and the thread is not suspended, returns None.
        """
    try:
        cmd_text = ['<xml><thread id="%s">' % (thread_id,)]
        if topmost_frame is not None:
            try:
                suspended_frames_manager = py_db.suspended_frames_manager
                frames_list = suspended_frames_manager.get_frames_list(thread_id)
                if frames_list is None:
                    if must_be_suspended:
                        return None
                    else:
                        frames_list = pydevd_frame_utils.create_frames_list_from_frame(topmost_frame)
                cmd_text.append(self.make_thread_stack_str(py_db, frames_list))
            finally:
                topmost_frame = None
        cmd_text.append('</thread></xml>')
        return NetCommand(CMD_GET_THREAD_STACK, seq, ''.join(cmd_text))
    except:
        return self.make_error_message(seq, get_exception_traceback_str())
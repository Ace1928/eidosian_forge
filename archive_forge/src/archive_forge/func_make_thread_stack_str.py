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
def make_thread_stack_str(self, py_db, frames_list):
    assert frames_list.__class__ == FramesList
    make_valid_xml_value = pydevd_xml.make_valid_xml_value
    cmd_text_list = []
    append = cmd_text_list.append
    try:
        for frame_id, frame, method_name, _original_filename, filename_in_utf8, lineno, _applied_mapping, _show_as_current_frame, line_col_info in self._iter_visible_frames_info(py_db, frames_list, flatten_chained=True):
            append('<frame id="%s" name="%s" ' % (frame_id, make_valid_xml_value(method_name)))
            append('file="%s" line="%s">' % (quote(make_valid_xml_value(filename_in_utf8), '/>_= \t'), lineno))
            append('</frame>')
    except:
        pydev_log.exception()
    return ''.join(cmd_text_list)
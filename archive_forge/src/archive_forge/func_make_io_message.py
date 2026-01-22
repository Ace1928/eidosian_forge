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
def make_io_message(self, msg, ctx):
    """
        @param msg: the message to pass to the debug server
        @param ctx: 1 for stdio 2 for stderr
        """
    try:
        msg = pydevd_constants.as_str(msg)
        if len(msg) > MAX_IO_MSG_SIZE:
            msg = msg[0:MAX_IO_MSG_SIZE]
            msg += '...'
        msg = pydevd_xml.make_valid_xml_value(quote(msg, '/>_= '))
        return NetCommand(str(CMD_WRITE_TO_CONSOLE), 0, '<xml><io s="%s" ctx="%s"/></xml>' % (msg, ctx))
    except:
        return self.make_error_message(0, get_exception_traceback_str())
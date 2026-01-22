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
def make_evaluation_timeout_msg(self, py_db, expression, thread):
    msg = "pydevd: Evaluating: %s did not finish after %.2f seconds.\nThis may mean a number of things:\n- This evaluation is really slow and this is expected.\n    In this case it's possible to silence this error by raising the timeout, setting the\n    PYDEVD_WARN_EVALUATION_TIMEOUT environment variable to a bigger value.\n\n- The evaluation may need other threads running while it's running:\n    In this case, you may need to manually let other paused threads continue.\n\n    Alternatively, it's also possible to skip breaking on a particular thread by setting a\n    `pydev_do_not_trace = True` attribute in the related threading.Thread instance\n    (if some thread should always be running and no breakpoints are expected to be hit in it).\n\n- The evaluation is deadlocked:\n    In this case you may set the PYDEVD_THREAD_DUMP_ON_WARN_EVALUATION_TIMEOUT\n    environment variable to true so that a thread dump is shown along with this message and\n    optionally, set the PYDEVD_INTERRUPT_THREAD_TIMEOUT to some value so that the debugger\n    tries to interrupt the evaluation (if possible) when this happens.\n" % (expression, pydevd_constants.PYDEVD_WARN_EVALUATION_TIMEOUT)
    if pydevd_constants.PYDEVD_THREAD_DUMP_ON_WARN_EVALUATION_TIMEOUT:
        stream = StringIO()
        pydevd_utils.dump_threads(stream, show_pydevd_threads=False)
        msg += '\n\n%s\n' % stream.getvalue()
    return self.make_warning_message(msg)
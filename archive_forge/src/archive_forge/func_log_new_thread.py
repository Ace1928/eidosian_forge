import time
from _pydev_bundle._pydev_filesystem_encoding import getfilesystemencoding
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle import pydevd_xml
from _pydevd_bundle.pydevd_constants import GlobalDebuggerHolder
from _pydevd_bundle.pydevd_constants import get_thread_id
from _pydevd_bundle.pydevd_net_command import NetCommand
from _pydevd_bundle.pydevd_concurrency_analyser.pydevd_thread_wrappers import ObjectWrapper, wrap_attr
import pydevd_file_utils
from _pydev_bundle import pydev_log
import sys
from urllib.parse import quote
def log_new_thread(global_debugger, t):
    event_time = cur_time() - global_debugger.thread_analyser.start_time
    send_concurrency_message('threading_event', event_time, t.name, get_thread_id(t), 'thread', 'start', 'code_name', 0, None, parent=get_thread_id(t))
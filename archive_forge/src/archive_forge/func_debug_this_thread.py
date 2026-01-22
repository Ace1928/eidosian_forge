import codecs
import os
import pydevd
import socket
import sys
import threading
import debugpy
from debugpy import adapter
from debugpy.common import json, log, sockets
from _pydevd_bundle.pydevd_constants import get_global_debugger
from pydevd_file_utils import absolute_path
from debugpy.common.util import hide_debugpy_internals
def debug_this_thread():
    ensure_logging()
    log.debug('debug_this_thread()')
    _settrace(suspend=False)
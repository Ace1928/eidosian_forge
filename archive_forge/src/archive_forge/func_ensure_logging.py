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
def ensure_logging():
    """Starts logging to log.log_dir, if it hasn't already been done."""
    if ensure_logging.ensured:
        return
    ensure_logging.ensured = True
    log.to_file(prefix='debugpy.server')
    log.describe_environment('Initial environment:')
    if log.log_dir is not None:
        pydevd.log_to(log.log_dir + '/debugpy.pydevd.log')
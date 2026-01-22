from __future__ import nested_scopes
import traceback
import warnings
from _pydev_bundle import pydev_log
from _pydev_bundle._pydev_saved_modules import thread, threading
from _pydev_bundle import _pydev_saved_modules
import signal
import os
import ctypes
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from urllib.parse import quote  # @UnresolvedImport
import time
import inspect
import sys
from _pydevd_bundle.pydevd_constants import USE_CUSTOM_SYS_CURRENT_FRAMES, IS_PYPY, SUPPORT_GEVENT, \

    Generates a KeyboardInterrupt in the main thread by sending a Ctrl+C
    or by calling thread.interrupt_main().

    :param main_thread:
        Needed because Jython needs main_thread._thread.interrupt() to be called.

    Note: if unable to send a Ctrl+C, the KeyboardInterrupt will only be raised
    when the next Python instruction is about to be executed (so, it won't interrupt
    a sleep(1000)).
    
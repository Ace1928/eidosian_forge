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
def interrupt_main_thread(main_thread=None):
    """
    Generates a KeyboardInterrupt in the main thread by sending a Ctrl+C
    or by calling thread.interrupt_main().

    :param main_thread:
        Needed because Jython needs main_thread._thread.interrupt() to be called.

    Note: if unable to send a Ctrl+C, the KeyboardInterrupt will only be raised
    when the next Python instruction is about to be executed (so, it won't interrupt
    a sleep(1000)).
    """
    if main_thread is None:
        main_thread = threading.main_thread()
    pydev_log.debug('Interrupt main thread.')
    called = False
    try:
        if os.name == 'posix':
            os.kill(os.getpid(), signal.SIGINT)
            called = True
        elif os.name == 'nt':
            ctypes.windll.kernel32.CtrlRoutine(0)
            called = True
    except:
        pydev_log.exception('Error interrupting main thread (using fallback).')
    if not called:
        try:
            if hasattr(thread, 'interrupt_main'):
                thread.interrupt_main()
            else:
                main_thread._thread.interrupt()
        except:
            pydev_log.exception('Error on interrupt main thread fallback.')
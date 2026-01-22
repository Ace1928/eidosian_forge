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
def notify_about_gevent_if_needed(stream=None):
    """
    When debugging with gevent check that the gevent flag is used if the user uses the gevent
    monkey-patching.

    :return bool:
        Returns True if a message had to be shown to the user and False otherwise.
    """
    stream = stream if stream is not None else sys.stderr
    if not SUPPORT_GEVENT:
        gevent_monkey = sys.modules.get('gevent.monkey')
        if gevent_monkey is not None:
            try:
                saved = gevent_monkey.saved
            except AttributeError:
                pydev_log.exception_once('Error checking for gevent monkey-patching.')
                return False
            if saved:
                sys.stderr.write('%s\n' % (GEVENT_SUPPORT_NOT_SET_MSG,))
                return True
    return False
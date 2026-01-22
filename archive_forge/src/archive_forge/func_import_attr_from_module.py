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
def import_attr_from_module(import_with_attr_access):
    if '.' not in import_with_attr_access:
        raise ImportError('Unable to import module with attr access: %s' % (import_with_attr_access,))
    module_name, attr_name = import_with_attr_access.rsplit('.', 1)
    while True:
        try:
            mod = import_module(module_name)
        except ImportError:
            if '.' not in module_name:
                raise ImportError('Unable to import module with attr access: %s' % (import_with_attr_access,))
            module_name, new_attr_part = module_name.rsplit('.', 1)
            attr_name = new_attr_part + '.' + attr_name
        else:
            try:
                for attr in attr_name.split('.'):
                    mod = getattr(mod, attr)
                return mod
            except:
                raise ImportError('Unable to import module with attr access: %s' % (import_with_attr_access,))
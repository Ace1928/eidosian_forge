from __future__ import nested_scopes
import platform
import weakref
import struct
import warnings
import functools
from contextlib import contextmanager
import sys  # Note: the sys import must be here anyways (others depend on it)
import codecs as _codecs
import os
from _pydevd_bundle import pydevd_vm_type
from _pydev_bundle._pydev_saved_modules import thread, threading
def protect_libraries_from_patching():
    """
    In this function we delete some modules from `sys.modules` dictionary and import them again inside
      `_pydev_saved_modules` in order to save their original copies there. After that we can use these
      saved modules within the debugger to protect them from patching by external libraries (e.g. gevent).
    """
    patched = ['threading', 'thread', '_thread', 'time', 'socket', 'queue', 'select', 'xmlrpclib', 'SimpleXMLRPCServer', 'BaseHTTPServer', 'SocketServer', 'xmlrpc.client', 'xmlrpc.server', 'http.server', 'socketserver']
    for name in patched:
        try:
            __import__(name)
        except:
            pass
    patched_modules = dict([(k, v) for k, v in sys.modules.items() if k in patched])
    for name in patched_modules:
        del sys.modules[name]
    import _pydev_bundle._pydev_saved_modules
    for name in patched_modules:
        sys.modules[name] = patched_modules[name]
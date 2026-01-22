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
def clear_cached_thread_id(thread):
    with _thread_id_lock:
        try:
            if thread.__pydevd_id__ != 'console_main':
                del thread.__pydevd_id__
        except AttributeError:
            pass
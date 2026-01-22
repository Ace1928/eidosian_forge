import ctypes
import logging
import os
import sys
from contextlib import contextmanager
from functools import partial
def get_libc():
    if IS_WINDOWS or IS_MACOS:
        logger.warning('NOTE: Redirects are currently not supported in Windows or MacOs.')
        return None
    else:
        return ctypes.CDLL('libc.so.6')
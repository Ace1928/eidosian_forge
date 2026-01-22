from __future__ import (absolute_import, division,
from future import utils
from future.builtins import str, range, open, int, map, list
import contextlib
import errno
import functools
import gc
import socket
import sys
import os
import platform
import shutil
import warnings
import unittest
import importlib
import re
import subprocess
import time
import fnmatch
import logging.handlers
import struct
import tempfile
def bigaddrspacetest(f):
    """Decorator for tests that fill the address space."""

    def wrapper(self):
        if max_memuse < MAX_Py_ssize_t:
            if MAX_Py_ssize_t >= 2 ** 63 - 1 and max_memuse >= 2 ** 31:
                raise unittest.SkipTest('not enough memory: try a 32-bit build instead')
            else:
                raise unittest.SkipTest('not enough memory: %.1fG minimum needed' % (MAX_Py_ssize_t / 1024 ** 3))
        else:
            return f(self)
    return wrapper
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
def filter_error(err):
    n = getattr(err, 'errno', None)
    if isinstance(err, socket.timeout) or (isinstance(err, socket.gaierror) and n in gai_errnos) or n in captured_errnos:
        if not verbose:
            sys.stderr.write(denied.args[0] + '\n')
        exc = denied
        exc.__cause__ = err
        raise exc
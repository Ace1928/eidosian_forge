import __future__
import builtins
import importlib._bootstrap
import importlib._bootstrap_external
import importlib.machinery
import importlib.util
import inspect
import io
import os
import pkgutil
import platform
import re
import sys
import sysconfig
import time
import tokenize
import urllib.parse
import warnings
from collections import deque
from reprlib import Repr
from traceback import format_exception_only
def pipepager(text, cmd):
    """Page through text by feeding it to another program."""
    import subprocess
    proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, errors='backslashreplace')
    try:
        with proc.stdin as pipe:
            try:
                pipe.write(text)
            except KeyboardInterrupt:
                pass
    except OSError:
        pass
    while True:
        try:
            proc.wait()
            break
        except KeyboardInterrupt:
            pass
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
def source_synopsis(file):
    line = file.readline()
    while line[:1] == '#' or not line.strip():
        line = file.readline()
        if not line:
            break
    line = line.strip()
    if line[:4] == 'r"""':
        line = line[1:]
    if line[:3] == '"""':
        line = line[3:]
        if line[-1:] == '\\':
            line = line[:-1]
        while not line.strip():
            line = file.readline()
            if not line:
                break
        result = line.split('"""')[0].strip()
    else:
        result = None
    return result
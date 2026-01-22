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
def repr_string(self, x, level):
    test = cram(x, self.maxstring)
    testrepr = repr(test)
    if '\\' in test and '\\' not in replace(testrepr, '\\\\', ''):
        return 'r' + testrepr[0] + test + testrepr[0]
    return testrepr
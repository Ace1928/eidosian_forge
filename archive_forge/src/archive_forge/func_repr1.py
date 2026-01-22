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
def repr1(self, x, level):
    if hasattr(type(x), '__name__'):
        methodname = 'repr_' + '_'.join(type(x).__name__.split())
        if hasattr(self, methodname):
            return getattr(self, methodname)(x, level)
    return cram(stripid(repr(x)), self.maxother)
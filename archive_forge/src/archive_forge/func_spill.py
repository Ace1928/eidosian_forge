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
def spill(msg, attrs, predicate):
    ok, attrs = _split_list(attrs, predicate)
    if ok:
        hr.maybe()
        push(msg)
        for name, kind, homecls, value in ok:
            try:
                value = getattr(object, name)
            except Exception:
                push(self.docdata(value, name, mod))
            else:
                push(self.document(value, name, mod, object))
    return attrs
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
def html_getobj(url):
    obj = locate(url, forceload=1)
    if obj is None and url != 'None':
        raise ValueError('could not find object')
    title = describe(obj)
    content = html.document(obj, url)
    return (title, content)
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
def html_index():
    """Module Index page."""

    def bltinlink(name):
        return '<a href="%s.html">%s</a>' % (name, name)
    heading = html.heading('<strong class="title">Index of Modules</strong>')
    names = [name for name in sys.builtin_module_names if name != '__main__']
    contents = html.multicolumn(names, bltinlink)
    contents = [heading, '<p>' + html.bigsection('Built-in Modules', 'index', contents)]
    seen = {}
    for dir in sys.path:
        contents.append(html.index(dir, seen))
    contents.append('<p align=right class="heading-text grey"><strong>pydoc</strong> by Ka-Ping Yee&lt;ping@lfw.org&gt;</p>')
    return ('Index of Modules', ''.join(contents))
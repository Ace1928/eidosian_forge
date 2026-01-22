from __future__ import annotations
import ast
import collections.abc as cabc
import importlib.metadata
import inspect
import os
import platform
import re
import sys
import traceback
import typing as t
from functools import update_wrapper
from operator import itemgetter
from types import ModuleType
import click
from click.core import ParameterSource
from werkzeug import run_simple
from werkzeug.serving import is_running_from_reloader
from werkzeug.utils import import_string
from .globals import current_app
from .helpers import get_debug_flag
from .helpers import get_load_dotenv
def prepare_import(path: str) -> str:
    """Given a filename this will try to calculate the python path, add it
    to the search path and return the actual module name that is expected.
    """
    path = os.path.realpath(path)
    fname, ext = os.path.splitext(path)
    if ext == '.py':
        path = fname
    if os.path.basename(path) == '__init__':
        path = os.path.dirname(path)
    module_name = []
    while True:
        path, name = os.path.split(path)
        module_name.append(name)
        if not os.path.exists(os.path.join(path, '__init__.py')):
            break
    if sys.path[0] != path:
        sys.path.insert(0, path)
    return '.'.join(module_name[::-1])
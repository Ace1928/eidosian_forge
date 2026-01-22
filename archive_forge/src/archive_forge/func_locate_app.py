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
def locate_app(module_name: str, app_name: str | None, raise_if_not_found: bool=True) -> Flask | None:
    try:
        __import__(module_name)
    except ImportError:
        if sys.exc_info()[2].tb_next:
            raise NoAppException(f'While importing {module_name!r}, an ImportError was raised:\n\n{traceback.format_exc()}') from None
        elif raise_if_not_found:
            raise NoAppException(f'Could not import {module_name!r}.') from None
        else:
            return None
    module = sys.modules[module_name]
    if app_name is None:
        return find_best_app(module)
    else:
        return find_app_by_string(module, app_name)
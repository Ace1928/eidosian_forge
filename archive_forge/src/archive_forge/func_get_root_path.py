from __future__ import annotations
import importlib.util
import os
import sys
import typing as t
from datetime import datetime
from functools import lru_cache
from functools import update_wrapper
import werkzeug.utils
from werkzeug.exceptions import abort as _wz_abort
from werkzeug.utils import redirect as _wz_redirect
from werkzeug.wrappers import Response as BaseResponse
from .globals import _cv_request
from .globals import current_app
from .globals import request
from .globals import request_ctx
from .globals import session
from .signals import message_flashed
def get_root_path(import_name: str) -> str:
    """Find the root path of a package, or the path that contains a
    module. If it cannot be found, returns the current working
    directory.

    Not to be confused with the value returned by :func:`find_package`.

    :meta private:
    """
    mod = sys.modules.get(import_name)
    if mod is not None and hasattr(mod, '__file__') and (mod.__file__ is not None):
        return os.path.dirname(os.path.abspath(mod.__file__))
    try:
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            raise ValueError
    except (ImportError, ValueError):
        loader = None
    else:
        loader = spec.loader
    if loader is None:
        return os.getcwd()
    if hasattr(loader, 'get_filename'):
        filepath = loader.get_filename(import_name)
    else:
        __import__(import_name)
        mod = sys.modules[import_name]
        filepath = getattr(mod, '__file__', None)
        if filepath is None:
            raise RuntimeError(f"No root path can be found for the provided module {import_name!r}. This can happen because the module came from an import hook that does not provide file name information or because it's a namespace package. In this case the root path needs to be explicitly provided.")
    return os.path.dirname(os.path.abspath(filepath))
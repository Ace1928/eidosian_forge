from __future__ import annotations
import logging # isort:skip
import importlib.util
from os.path import isfile
from types import ModuleType
from typing import (
from tornado.httputil import HTTPServerRequest
from tornado.web import RequestHandler
from ..util.serialization import make_globally_unique_id
def load_auth_module(module_path: PathLike) -> ModuleType:
    """ Load a Python source file at a given path as a module.

    Arguments:
        module_path (str): path to a Python source file

    Returns
        module

    """
    module_name = 'bokeh.auth_' + make_globally_unique_id().replace('-', '')
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
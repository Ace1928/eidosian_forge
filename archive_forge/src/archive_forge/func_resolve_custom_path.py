from __future__ import annotations
import functools
import importlib
import json
import logging
import mimetypes
import os
import pathlib
import re
import textwrap
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import (
import param
from bokeh.embed.bundle import (
from bokeh.model import Model
from bokeh.models import ImportedStyleSheet
from bokeh.resources import Resources as BkResources, _get_server_urls
from bokeh.settings import settings as _settings
from jinja2.environment import Environment
from jinja2.loaders import FileSystemLoader
from markupsafe import Markup
from ..config import config, panel_extension as extension
from ..util import isurl, url_path
from .state import state
def resolve_custom_path(obj, path: str | os.PathLike, relative: bool=False) -> pathlib.Path | None:
    """
    Attempts to resolve a path relative to some component.

    Arguments
    ---------
    obj: type | object
       The component to resolve the path relative to.
    path: str | os.PathLike
        Absolute or relative path to a resource.
    relative: bool
        Whether to return a relative path.

    Returns
    -------
    path: pathlib.Path | None
    """
    if not path:
        return
    if not isinstance(obj, type):
        obj = type(obj)
    try:
        mod = importlib.import_module(obj.__module__)
        module_path = Path(mod.__file__).parent
        assert module_path.exists()
    except Exception:
        return None
    path = pathlib.Path(path)
    if path.is_absolute():
        abs_path = path
    else:
        abs_path = module_path / path
    try:
        if not abs_path.is_file():
            return None
    except OSError:
        return None
    abs_path = abs_path.resolve()
    if not relative:
        return abs_path
    return os.path.relpath(abs_path, module_path)
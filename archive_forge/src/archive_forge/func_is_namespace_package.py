from __future__ import annotations
import errno
import importlib.util
import os
import socket
import sys
import warnings
from contextlib import contextmanager
from typing import Any, Generator, NewType, Sequence
from urllib.parse import (
from urllib.parse import (
from urllib.request import pathname2url as _pathname2url
from _frozen_importlib_external import _NamespacePath
from jupyter_core.utils import ensure_async as _ensure_async
from packaging.version import Version
from tornado.httpclient import AsyncHTTPClient, HTTPClient, HTTPRequest, HTTPResponse
from tornado.netutil import Resolver
def is_namespace_package(namespace: str) -> bool | None:
    """Is the provided namespace a Python Namespace Package (PEP420).

    https://www.python.org/dev/peps/pep-0420/#specification

    Returns `None` if module is not importable.

    """
    try:
        spec = importlib.util.find_spec(namespace)
    except ValueError:
        return None
    if not spec:
        return None
    return isinstance(spec.submodule_search_locations, _NamespacePath)
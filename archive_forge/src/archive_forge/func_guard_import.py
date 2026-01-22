import contextlib
import datetime
import functools
import importlib
import warnings
from importlib.metadata import version
from typing import Any, Callable, Dict, Optional, Set, Tuple, Union
from packaging.version import parse
from requests import HTTPError, Response
from langchain_core.pydantic_v1 import SecretStr
def guard_import(module_name: str, *, pip_name: Optional[str]=None, package: Optional[str]=None) -> Any:
    """Dynamically imports a module and raises a helpful exception if the module is not
    installed."""
    try:
        module = importlib.import_module(module_name, package)
    except ImportError:
        raise ImportError(f'Could not import {module_name} python package. Please install it with `pip install {pip_name or module_name}`.')
    return module
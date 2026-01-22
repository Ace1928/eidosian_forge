from __future__ import annotations
import logging # isort:skip
from importlib import import_module
from types import ModuleType
from typing import Any
def import_required(mod_name: str, error_msg: str) -> ModuleType:
    """ Attempt to import a required dependency.

    Raises a RuntimeError if the requested module is not available.

    Args:
        mod_name (str) : name of the required module to try to import
        error_msg (str) : error message to raise when the module is missing

    Returns:
        imported module

    Raises:
        RuntimeError

    """
    try:
        return import_module(mod_name)
    except ImportError as e:
        raise RuntimeError(error_msg) from e
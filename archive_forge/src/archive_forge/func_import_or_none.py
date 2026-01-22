from __future__ import annotations
import contextlib
import importlib
import time
from typing import TYPE_CHECKING
def import_or_none(name):
    """Import a module and return it; in case of failure; return None"""
    try:
        return importlib.import_module(name)
    except (ImportError, AttributeError):
        return None
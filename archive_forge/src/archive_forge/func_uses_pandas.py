from __future__ import annotations
import logging # isort:skip
from importlib import import_module
from types import ModuleType
from typing import Any
def uses_pandas(obj: Any) -> bool:
    """
    Checks if an object is a ``pandas`` object.

    Use this before conditional ``import pandas as pd``.
    """
    module = type(obj).__module__
    return module is not None and module.startswith('pandas.')
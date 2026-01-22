from __future__ import annotations
import sys
import copy
import pathlib
import inspect
import functools
import importlib.util
from typing import Any, Dict, Callable, Union, Optional, Type, TypeVar, List, Tuple, cast, TYPE_CHECKING
from types import ModuleType
def get_keydb_enabled() -> bool:
    """
    Gets whether or not keydb is enabled
    """
    if _keydb_enabled is None:
        get_keydb_session(validate_active=True)
    return _keydb_enabled
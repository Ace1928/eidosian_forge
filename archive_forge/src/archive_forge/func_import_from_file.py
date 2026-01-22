from __future__ import annotations
import sys
import copy
import pathlib
import inspect
import functools
import importlib.util
from typing import Any, Dict, Callable, Union, Optional, Type, TypeVar, List, Tuple, cast, TYPE_CHECKING
from types import ModuleType
def import_from_file(file: Union[str, pathlib.Path], cls_name: str) -> ModuleType:
    """
    Import a file
    """
    return load_module_from_file(file=file, module_name=cls_name)
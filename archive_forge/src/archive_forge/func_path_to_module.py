from __future__ import annotations
import ast
import os
import re
import typing as t
from ..io import (
from ..util import (
from ..data import (
from ..target import (
def path_to_module(path: str) -> str:
    """Convert the given path to a module name."""
    module = os.path.splitext(path)[0].replace(os.path.sep, '.')
    if module.endswith('.__init__'):
        module = module[:-9]
    return module
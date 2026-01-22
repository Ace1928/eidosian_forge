from __future__ import annotations
import ast
import os
import re
import typing as t
from ..io import (
from ..util import (
from ..data import (
from ..target import (
def relative_to_absolute(name: str, level: int, module: str, path: str, lineno: int) -> str:
    """Convert a relative import to an absolute import."""
    if level <= 0:
        absolute_name = name
    elif not module:
        display.warning('Cannot resolve relative import "%s%s" in unknown module at %s:%d' % ('.' * level, name, path, lineno))
        absolute_name = 'relative.nomodule'
    else:
        parts = module.split('.')
        if level >= len(parts):
            display.warning('Cannot resolve relative import "%s%s" above module "%s" at %s:%d' % ('.' * level, name, module, path, lineno))
            absolute_name = 'relative.abovelevel'
        else:
            absolute_name = '.'.join(parts[:-level] + [name])
    return absolute_name
from __future__ import annotations
from collections.abc import Callable, Mapping
from typing import TypeVar
from sys import stdout
def debug_rl(*args, **kwargs):
    expr = args[0]
    result = rule(*args, **kwargs)
    if result != expr:
        file.write('Rule: %s\n' % rule.__name__)
        file.write('In:   %s\nOut:  %s\n\n' % (expr, result))
    return result
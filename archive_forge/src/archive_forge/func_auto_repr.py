import inspect
from functools import partial
from typing import (
def auto_repr(self: T) -> str:
    """Create repr string from __rich_repr__"""
    repr_str: List[str] = []
    append = repr_str.append
    angular: bool = getattr(self.__rich_repr__, 'angular', False)
    for arg in self.__rich_repr__():
        if isinstance(arg, tuple):
            if len(arg) == 1:
                append(repr(arg[0]))
            else:
                key, value, *default = arg
                if key is None:
                    append(repr(value))
                else:
                    if default and default[0] == value:
                        continue
                    append(f'{key}={value!r}')
        else:
            append(repr(arg))
    if angular:
        return f'<{self.__class__.__name__} {' '.join(repr_str)}>'
    else:
        return f'{self.__class__.__name__}({', '.join(repr_str)})'
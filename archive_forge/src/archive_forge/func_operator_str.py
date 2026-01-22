from __future__ import annotations
import os
import tempfile
import textwrap
from functools import lru_cache
@staticmethod
def operator_str(target, args, kwargs):
    lines = [f'target: {target}'] + [f'args[{i}]: {arg}' for i, arg in enumerate(args)]
    if kwargs:
        lines.append(f'kwargs: {kwargs}')
    return textwrap.indent('\n'.join(lines), '  ')
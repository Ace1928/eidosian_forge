from __future__ import annotations
import collections
import collections.abc
import string
from collections.abc import Sequence
import numpy as np
@staticmethod
def format_list2d(values, float_decimal=0):
    """Format a list of lists."""
    flattened_list = flatten(values)
    if all((isinstance(v, int) for v in flattened_list)):
        type_all = int
    else:
        try:
            for v in flattened_list:
                float(v)
            type_all = float
        except Exception:
            type_all = str
    width = max((len(str(s)) for s in flattened_list))
    if type_all is int:
        fmt_spec = f'>{width}d'
    elif type_all is str:
        fmt_spec = f'>{width}'
    else:
        max_dec = max((len(str(f - int(f))) - 2 for f in flattened_list))
        n_dec = min(max(max_dec, float_decimal), 10)
        if all((f == 0 or (abs(f) > 0.001 and abs(f) < 10000.0) for f in flattened_list)):
            fmt_spec = f'>{n_dec + 5}.{n_dec}f'
        else:
            fmt_spec = f'>{n_dec + 8}.{n_dec}e'
    line = '\n'
    for lst in values:
        for val in lst:
            line += f' {val:{ {fmt_spec}}}'
        line += '\n'
    return line.rstrip('\n')
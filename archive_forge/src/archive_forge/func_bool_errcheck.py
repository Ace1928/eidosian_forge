from __future__ import annotations
import sys
def bool_errcheck(result, func, args):
    if not result:
        raise WinError()
    return args
from __future__ import annotations
from contextlib import (
import re
from typing import (
import warnings
from pandas._typing import (
from pandas.util._exceptions import find_stack_level
def is_one_of_factory(legal_values) -> Callable[[Any], None]:
    callables = [c for c in legal_values if callable(c)]
    legal_values = [c for c in legal_values if not callable(c)]

    def inner(x) -> None:
        if x not in legal_values:
            if not any((c(x) for c in callables)):
                uvals = [str(lval) for lval in legal_values]
                pp_values = '|'.join(uvals)
                msg = f'Value must be one of {pp_values}'
                if len(callables):
                    msg += ' or a callable'
                raise ValueError(msg)
    return inner
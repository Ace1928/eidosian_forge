from __future__ import annotations
import decimal
import struct
from typing import Any, Sequence, Tuple, Type, Union
def create_decimal128_context() -> decimal.Context:
    """Returns an instance of :class:`decimal.Context` appropriate
    for working with IEEE-754 128-bit decimal floating point values.
    """
    opts = _CTX_OPTIONS.copy()
    opts['traps'] = []
    return decimal.Context(**opts)
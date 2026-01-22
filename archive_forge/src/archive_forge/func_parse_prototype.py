import re
import warnings
from numba.core import typing, sigutils
from numba.pycc.compiler import ExportEntry
def parse_prototype(text):
    """Separate the symbol and function-type in a a string with
    "symbol function-type" (e.g. "mult float(float, float)")

    Returns
    ---------
    (symbol_string, functype_string)
    """
    m = re_symbol.match(text)
    if not m:
        raise ValueError('Invalid function name for export prototype')
    s = m.start(0)
    e = m.end(0)
    symbol = text[s:e]
    functype = text[e + 1:]
    return (symbol, functype)
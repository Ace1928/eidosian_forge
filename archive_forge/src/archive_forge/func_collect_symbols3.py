import pytest
from numpy.f2py.symbolic import (
from . import util
def collect_symbols3(expr, symbols):
    if expr.op is Op.APPLY:
        return expr
    if expr.op is Op.SYMBOL:
        symbols.add(expr)
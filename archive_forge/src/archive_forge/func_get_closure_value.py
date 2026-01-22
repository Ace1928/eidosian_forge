import dis
from contextlib import contextmanager
import builtins
import operator
from typing import Iterator
from functools import reduce
from numba.core import (
from numba.core.utils import (
from .rvsdg.bc2rvsdg import (
from .rvsdg.regionpasses import RegionVisitor
def get_closure_value(self, index):
    """
        Get a value from the cell contained in this function's closure.
        If not set, return a ir.UNDEFINED.
        """
    cell = self.func_id.func.__closure__[index]
    try:
        return cell.cell_contents
    except ValueError:
        return ir.UNDEFINED
from collections import OrderedDict
from functools import reduce
import math
from operator import add
from ..units import get_derived_unit, default_units, energy, concentration
from ..util._dimensionality import dimension_codes, base_registry
from ..util.pyutil import memoize, deprecated
from ..util._expr import Expr, UnaryWrapper, Symbol
def get_named_keys(self):
    arg, = self.args
    if isinstance(arg, Symbol):
        return arg.args
    else:
        return self.unique_keys
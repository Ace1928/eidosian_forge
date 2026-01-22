from collections import OrderedDict, defaultdict
from functools import reduce
from itertools import chain, product
from operator import mul, add
import copy
import math
import warnings
from .util.arithmeticdict import ArithmeticDict
from .util._expr import Expr
from .util.periodic import mass_from_composition
from .util.parsing import (
from .units import default_units, is_quantity, unit_of, to_unitless
from ._util import intdiv
from .util.pyutil import deprecated, DeferredImport, ChemPyDeprecationWarning
def check_consistent_units(self, throw=False):
    if is_quantity(self.param):
        exponent = sum(self.prod.values()) - sum(self.reac.values())
        unit_param = unit_of(self.param, simplified=True)
        unit_expected = unit_of(default_units.molar ** exponent, simplified=True)
        if unit_param == unit_expected:
            return True
        elif throw:
            raise ValueError('Inconsistent units in equilibrium: %s vs %s' % (unit_param, unit_expected))
        else:
            return False
    else:
        return True
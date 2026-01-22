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
def check_any_effect(self, throw=False):
    """Checks if the reaction has any effect"""
    if not any(self.net_stoich(self.keys())):
        if throw:
            raise ValueError('The net stoichiometry change of all species are zero.')
        else:
            return False
    return True
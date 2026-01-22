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
@staticmethod
def composition_keys(substance_iter, skip_keys=()):
    """Occurring :attr:`composition` keys among a series of substances"""
    keys = set()
    for s in substance_iter:
        if s.composition is None:
            continue
        for k in s.composition.keys():
            if k in skip_keys:
                continue
            keys.add(k)
    return sorted(keys)
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
@property
@deprecated(last_supported_version='0.3.0', will_be_missing_in='0.8.0')
def precipitate(self):
    """deprecated attribute, provided for compatibility for now"""
    return self.phase_idx > 0
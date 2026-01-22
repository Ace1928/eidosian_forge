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
def molar_mass(self, units=None):
    """Returns the molar mass (with units) of the substance

        Examples
        --------
        >>> nh4p = Substance.from_formula('NH4+')  # simpler
        >>> from chempy.units import default_units as u
        >>> nh4p.molar_mass(u)
        array(18.0384511...) * g/mol

        """
    if units is None:
        units = default_units
    return self.mass * units.g / units.mol
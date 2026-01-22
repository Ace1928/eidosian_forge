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
def mass_fractions(stoichiometries, substances=None, substance_factory=Substance.from_formula):
    """Calculates weight fractions of each substance in a stoichiometric dict

    Parameters
    ----------
    stoichiometries : dict or set
        If a ``set``: all entries are assumed to correspond to unit multiplicity.
    substances: dict or None

    Examples
    --------
    >>> r = mass_fractions({'H2': 1, 'O2': 1})
    >>> mH2, mO2 = 1.008*2, 15.999*2
    >>> abs(r['H2'] - mH2/(mH2+mO2)) < 1e-4
    True
    >>> abs(r['O2'] - mO2/(mH2+mO2)) < 1e-4
    True
    >>> mass_fractions({'H2O2'}) == {'H2O2': 1.0}
    True

    """
    if isinstance(stoichiometries, set):
        stoichiometries = {k: 1 for k in stoichiometries}
    if substances is None:
        substances = OrderedDict([(k, substance_factory(k)) for k in stoichiometries])
    tot_mass = sum([substances[k].mass * v for k, v in stoichiometries.items()])
    return {k: substances[k].mass * v / tot_mass for k, v in stoichiometries.items()}
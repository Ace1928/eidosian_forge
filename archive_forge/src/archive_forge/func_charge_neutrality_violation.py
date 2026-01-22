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
def charge_neutrality_violation(self, substances):
    """Net amount of charge produced

        Parameters
        ----------
        substances: dict

        Returns
        -------
        float: amount of net charge produced/consumed

        """
    return self._violation(substances, 'charge')
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
def composition_violation(self, substances, composition_keys=None):
    """Net amount of constituent produced

        If composition keys correspond to conserved entities e.g. atoms
        in chemical reactions, this function should return a list of zeros.

        Parameters
        ----------
        substances : dict
        composition_keys : iterable of str, ``None`` or ``True``
            When ``None`` or True: composition keys are taken from substances.
            When ``True`` the keys are also return as an extra return value

        Returns
        -------
        - If ``composition_keys == True``: a tuple: (violations, composition_keys)
        - Otherwise: violations (list of coefficients)

        """
    keys, values = zip(*substances.items())
    ret_comp_keys = composition_keys is True
    if composition_keys in (None, True):
        composition_keys = Substance.composition_keys(values)
    net = [0] * len(composition_keys)
    for substance, coeff in zip(values, self.net_stoich(keys)):
        for idx, key in enumerate(composition_keys):
            net[idx] += substance.composition.get(key, 0) * coeff
    if ret_comp_keys:
        return (net, composition_keys)
    else:
        return net
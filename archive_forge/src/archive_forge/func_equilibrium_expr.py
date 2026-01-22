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
def equilibrium_expr(self):
    """Turns self.param into a :class:`EqExpr` instance (if not already)

        Examples
        --------
        >>> r = Equilibrium.from_string('2 A + B = 3 C; 7')
        >>> eqex = r.equilibrium_expr()
        >>> eqex.args[0] == 7
        True

        """
    from .util._expr import Expr
    from .thermodynamics import MassActionEq
    if isinstance(self.param, Expr):
        return self.param
    else:
        try:
            convertible = self.param.as_EqExpr
        except AttributeError:
            return MassActionEq([self.param])
        else:
            return convertible()
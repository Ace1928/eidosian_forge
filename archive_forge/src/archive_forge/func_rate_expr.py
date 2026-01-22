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
def rate_expr(self):
    """Turns self.param into a RateExpr instance (if not already)

        Default is to create a ``MassAction`` instance. The parameter will
        be used as single instance in ``unique_keys`` if it is a string,
        otherwise it will be used as ``args``.

        Examples
        --------
        >>> r = Reaction.from_string('2 A + B -> 3 C; 7')
        >>> ratex = r.rate_expr()
        >>> ratex.args[0] == 7
        True

        """
    from .util._expr import Expr
    from .kinetics import MassAction
    if isinstance(self.param, Expr):
        return self.param
    else:
        try:
            convertible = self.param.as_RateExpr
        except AttributeError:
            if isinstance(self.param, str):
                return MassAction.fk(self.param)
            else:
                return MassAction([self.param])
        else:
            return convertible()
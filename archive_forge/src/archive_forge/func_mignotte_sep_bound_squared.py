from sympy.core.numbers import oo
from sympy.core.sympify import CantSympify
from sympy.polys.polyerrors import CoercionFailed, NotReversible, NotInvertible
from sympy.polys.polyutils import PicklableWithSlots
from sympy.polys.densebasic import (
from sympy.polys.densearith import (
from sympy.polys.densetools import (
from sympy.polys.euclidtools import (
from sympy.polys.sqfreetools import (
from sympy.polys.factortools import (
from sympy.polys.rootisolation import (
from sympy.polys.polyerrors import (
def mignotte_sep_bound_squared(f):
    """Computes the squared Mignotte bound on root separations of ``f``. """
    if not f.lev:
        return dup_mignotte_sep_bound_squared(f.rep, f.dom)
    else:
        raise ValueError('univariate polynomial expected')
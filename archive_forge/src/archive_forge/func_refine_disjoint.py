from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.densetools import (
from sympy.polys.euclidtools import (
from sympy.polys.factortools import (
from sympy.polys.polyerrors import (
from sympy.polys.sqfreetools import (
def refine_disjoint(self, other):
    """Refine an isolating interval until it is disjoint with another one. """
    expr = self
    while not expr.is_disjoint(other):
        expr, other = (expr._inner_refine(), other._inner_refine())
    return (expr, other)
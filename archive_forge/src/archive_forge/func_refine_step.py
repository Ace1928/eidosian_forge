from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.densetools import (
from sympy.polys.euclidtools import (
from sympy.polys.factortools import (
from sympy.polys.polyerrors import (
from sympy.polys.sqfreetools import (
def refine_step(self, steps=1):
    """Perform several steps of complex root refinement algorithm. """
    expr = self
    for _ in range(steps):
        expr = expr._inner_refine()
    return expr
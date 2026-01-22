from sympy.core.expr import Expr
from sympy.core.symbol import Dummy
from sympy.core.sympify import _sympify
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polytools import Poly, parallel_poly_from_expr
from sympy.polys.domains import QQ
from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.domainscalar import DomainScalar
@classmethod
def from_dm(cls, dm):
    obj = super().__new__(cls)
    dm = dm.to_sparse()
    R = dm.domain
    obj._dm = dm
    obj.ring = R
    obj.domain = R.domain
    obj.gens = R.symbols
    return obj
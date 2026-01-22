from sympy.polys.domains.ring import Ring
from sympy.polys.polyerrors import NotReversible, DomainError
from sympy.utilities import public
def is_unit(self, a):
    """Return true if ``a`` is a invertible"""
    return bool(a)
from sympy.concrete.expr_with_limits import ExprWithLimits
from sympy.core.singleton import S
from sympy.core.relational import Eq
@property
def has_empty_sequence(self):
    """
        Returns True if the Sum or Product is computed for an empty sequence.

        Examples
        ========

        >>> from sympy import Sum, Product, Symbol
        >>> m = Symbol('m')
        >>> Sum(m, (m, 1, 0)).has_empty_sequence
        True

        >>> Sum(m, (m, 1, 1)).has_empty_sequence
        False

        >>> M = Symbol('M', integer=True, positive=True)
        >>> Product(m, (m, 1, M)).has_empty_sequence
        False

        >>> Product(m, (m, 2, M)).has_empty_sequence

        >>> Product(m, (m, M + 1, M)).has_empty_sequence
        True

        >>> N = Symbol('N', integer=True, positive=True)
        >>> Sum(m, (m, N, M)).has_empty_sequence

        >>> N = Symbol('N', integer=True, negative=True)
        >>> Sum(m, (m, N, M)).has_empty_sequence
        False

        See Also
        ========

        has_reversed_limits
        has_finite_limits

        """
    ret_None = False
    for lim in self.limits:
        dif = lim[1] - lim[2]
        eq = Eq(dif, 1)
        if eq == True:
            return True
        elif eq == False:
            continue
        else:
            ret_None = True
    if ret_None:
        return None
    return False
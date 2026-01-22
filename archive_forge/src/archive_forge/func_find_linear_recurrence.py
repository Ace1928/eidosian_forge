from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.decorators import call_highest_priority
from sympy.core.parameters import global_parameters
from sympy.core.function import AppliedUndef, expand
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.relational import Eq
from sympy.core.singleton import S, Singleton
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, Symbol, Wild
from sympy.core.sympify import sympify
from sympy.matrices import Matrix
from sympy.polys import lcm, factor
from sympy.sets.sets import Interval, Intersection
from sympy.tensor.indexed import Idx
from sympy.utilities.iterables import flatten, is_sequence, iterable
def find_linear_recurrence(self, n, d=None, gfvar=None):
    """
        Finds the shortest linear recurrence that satisfies the first n
        terms of sequence of order `\\leq` ``n/2`` if possible.
        If ``d`` is specified, find shortest linear recurrence of order
        `\\leq` min(d, n/2) if possible.
        Returns list of coefficients ``[b(1), b(2), ...]`` corresponding to the
        recurrence relation ``x(n) = b(1)*x(n-1) + b(2)*x(n-2) + ...``
        Returns ``[]`` if no recurrence is found.
        If gfvar is specified, also returns ordinary generating function as a
        function of gfvar.

        Examples
        ========

        >>> from sympy import sequence, sqrt, oo, lucas
        >>> from sympy.abc import n, x, y
        >>> sequence(n**2).find_linear_recurrence(10, 2)
        []
        >>> sequence(n**2).find_linear_recurrence(10)
        [3, -3, 1]
        >>> sequence(2**n).find_linear_recurrence(10)
        [2]
        >>> sequence(23*n**4+91*n**2).find_linear_recurrence(10)
        [5, -10, 10, -5, 1]
        >>> sequence(sqrt(5)*(((1 + sqrt(5))/2)**n - (-(1 + sqrt(5))/2)**(-n))/5).find_linear_recurrence(10)
        [1, 1]
        >>> sequence(x+y*(-2)**(-n), (n, 0, oo)).find_linear_recurrence(30)
        [1/2, 1/2]
        >>> sequence(3*5**n + 12).find_linear_recurrence(20,gfvar=x)
        ([6, -5], 3*(5 - 21*x)/((x - 1)*(5*x - 1)))
        >>> sequence(lucas(n)).find_linear_recurrence(15,gfvar=x)
        ([1, 1], (x - 2)/(x**2 + x - 1))
        """
    from sympy.simplify import simplify
    x = [simplify(expand(t)) for t in self[:n]]
    lx = len(x)
    if d is None:
        r = lx // 2
    else:
        r = min(d, lx // 2)
    coeffs = []
    for l in range(1, r + 1):
        l2 = 2 * l
        mlist = []
        for k in range(l):
            mlist.append(x[k:k + l])
        m = Matrix(mlist)
        if m.det() != 0:
            y = simplify(m.LUsolve(Matrix(x[l:l2])))
            if lx == l2:
                coeffs = flatten(y[::-1])
                break
            mlist = []
            for k in range(l, lx - l):
                mlist.append(x[k:k + l])
            m = Matrix(mlist)
            if m * y == Matrix(x[l2:]):
                coeffs = flatten(y[::-1])
                break
    if gfvar is None:
        return coeffs
    else:
        l = len(coeffs)
        if l == 0:
            return ([], None)
        else:
            n, d = (x[l - 1] * gfvar ** (l - 1), 1 - coeffs[l - 1] * gfvar ** l)
            for i in range(l - 1):
                n += x[i] * gfvar ** i
                for j in range(l - i - 1):
                    n -= coeffs[i] * x[j] * gfvar ** (i + j + 1)
                d -= coeffs[i] * gfvar ** (i + 1)
            return (coeffs, simplify(factor(n) / factor(d)))
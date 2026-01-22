import itertools
from functools import reduce
from sympy.core import Dummy, ilcm, Add, Mul, Pow, S
from sympy.integrals.rde import (order_at, order_at_oo, weak_normalizer,
from sympy.integrals.risch import (gcdex_diophantine, frac_in, derivation,
from sympy.polys import Poly, lcm, cancel, sqf_list
from sympy.polys.polymatrix import PolyMatrix as Matrix
from sympy.solvers import solve

    Checks if f can be written as the logarithmic derivative of a k(t)-radical.

    Explanation
    ===========

    It differs from is_log_deriv_k_t_radical(fa, fd, DE, Df=False)
    for any given fa, fd, DE in that it finds the solution in the
    given field not in some (possibly unspecified extension) and
    "in_field" with the function name is used to indicate that.

    f in k(t) can be written as the logarithmic derivative of a k(t) radical if
    there exist n in ZZ and u in k(t) with n, u != 0 such that n*f == Du/u.
    Either returns (n, u) or None, which means that f cannot be written as the
    logarithmic derivative of a k(t)-radical.

    case is one of {'primitive', 'exp', 'tan', 'auto'} for the primitive,
    hyperexponential, and hypertangent cases, respectively.  If case is 'auto',
    it will attempt to determine the type of the derivation automatically.

    See also
    ========
    is_log_deriv_k_t_radical, is_deriv_k

    
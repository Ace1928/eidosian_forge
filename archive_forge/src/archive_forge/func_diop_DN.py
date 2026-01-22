from sympy.core.add import Add
from sympy.core.assumptions import check_assumptions
from sympy.core.containers import Tuple
from sympy.core.exprtools import factor_terms
from sympy.core.function import _mexpand
from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.numbers import igcdex, ilcm, igcd
from sympy.core.power import integer_nthroot, isqrt
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Symbol, symbols
from sympy.core.sympify import _sympify
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.ntheory.factor_ import (
from sympy.ntheory.generate import nextprime
from sympy.ntheory.primetest import is_square, isprime
from sympy.ntheory.residue_ntheory import sqrt_mod
from sympy.polys.polyerrors import GeneratorsNeeded
from sympy.polys.polytools import Poly, factor_list
from sympy.simplify.simplify import signsimp
from sympy.solvers.solveset import solveset_real
from sympy.utilities import numbered_symbols
from sympy.utilities.misc import as_int, filldedent
from sympy.utilities.iterables import (is_sequence, subsets, permute_signs,
def diop_DN(D, N, t=symbols('t', integer=True)):
    """
    Solves the equation `x^2 - Dy^2 = N`.

    Explanation
    ===========

    Mainly concerned with the case `D > 0, D` is not a perfect square,
    which is the same as the generalized Pell equation. The LMM
    algorithm [1]_ is used to solve this equation.

    Returns one solution tuple, (`x, y)` for each class of the solutions.
    Other solutions of the class can be constructed according to the
    values of ``D`` and ``N``.

    Usage
    =====

    ``diop_DN(D, N, t)``: D and N are integers as in `x^2 - Dy^2 = N` and
    ``t`` is the parameter to be used in the solutions.

    Details
    =======

    ``D`` and ``N`` correspond to D and N in the equation.
    ``t`` is the parameter to be used in the solutions.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import diop_DN
    >>> diop_DN(13, -4) # Solves equation x**2 - 13*y**2 = -4
    [(3, 1), (393, 109), (36, 10)]

    The output can be interpreted as follows: There are three fundamental
    solutions to the equation `x^2 - 13y^2 = -4` given by (3, 1), (393, 109)
    and (36, 10). Each tuple is in the form (x, y), i.e. solution (3, 1) means
    that `x = 3` and `y = 1`.

    >>> diop_DN(986, 1) # Solves equation x**2 - 986*y**2 = 1
    [(49299, 1570)]

    See Also
    ========

    find_DN(), diop_bf_DN()

    References
    ==========

    .. [1] Solving the generalized Pell equation x**2 - D*y**2 = N, John P.
        Robertson, July 31, 2004, Pages 16 - 17. [online], Available:
        https://web.archive.org/web/20160323033128/http://www.jpr2718.org/pell.pdf
    """
    if D < 0:
        if N == 0:
            return [(0, 0)]
        elif N < 0:
            return []
        elif N > 0:
            sol = []
            for d in divisors(square_factor(N)):
                sols = cornacchia(1, -D, N // d ** 2)
                if sols:
                    for x, y in sols:
                        sol.append((d * x, d * y))
                        if D == -1:
                            sol.append((d * y, d * x))
            return sol
    elif D == 0:
        if N < 0:
            return []
        if N == 0:
            return [(0, t)]
        sN, _exact = integer_nthroot(N, 2)
        if _exact:
            return [(sN, t)]
        else:
            return []
    else:
        sD, _exact = integer_nthroot(D, 2)
        if _exact:
            if N == 0:
                return [(sD * t, t)]
            else:
                sol = []
                for y in range(floor(sign(N) * (N - 1) / (2 * sD)) + 1):
                    try:
                        sq, _exact = integer_nthroot(D * y ** 2 + N, 2)
                    except ValueError:
                        _exact = False
                    if _exact:
                        sol.append((sq, y))
                return sol
        elif 1 < N ** 2 < D:
            return _special_diop_DN(D, N)
        elif N == 0:
            return [(0, 0)]
        elif abs(N) == 1:
            pqa = PQa(0, 1, D)
            j = 0
            G = []
            B = []
            for i in pqa:
                a = i[2]
                G.append(i[5])
                B.append(i[4])
                if j != 0 and a == 2 * sD:
                    break
                j = j + 1
            if _odd(j):
                if N == -1:
                    x = G[j - 1]
                    y = B[j - 1]
                else:
                    count = j
                    while count < 2 * j - 1:
                        i = next(pqa)
                        G.append(i[5])
                        B.append(i[4])
                        count += 1
                    x = G[count]
                    y = B[count]
            elif N == 1:
                x = G[j - 1]
                y = B[j - 1]
            else:
                return []
            return [(x, y)]
        else:
            fs = []
            sol = []
            div = divisors(N)
            for d in div:
                if divisible(N, d ** 2):
                    fs.append(d)
            for f in fs:
                m = N // f ** 2
                zs = sqrt_mod(D, abs(m), all_roots=True)
                zs = [i for i in zs if i <= abs(m) // 2]
                if abs(m) != 2:
                    zs = zs + [-i for i in zs if i]
                for z in zs:
                    pqa = PQa(z, abs(m), D)
                    j = 0
                    G = []
                    B = []
                    for i in pqa:
                        G.append(i[5])
                        B.append(i[4])
                        if j != 0 and abs(i[1]) == 1:
                            r = G[j - 1]
                            s = B[j - 1]
                            if r ** 2 - D * s ** 2 == m:
                                sol.append((f * r, f * s))
                            elif diop_DN(D, -1) != []:
                                a = diop_DN(D, -1)
                                sol.append((f * (r * a[0][0] + a[0][1] * s * D), f * (r * a[0][1] + s * a[0][0])))
                            break
                        j = j + 1
                        if j == length(z, abs(m), D):
                            break
            return sol
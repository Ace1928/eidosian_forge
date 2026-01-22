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
def gaussian_reduce(w, a, b):
    """
    Returns a reduced solution `(x, z)` to the congruence
    `X^2 - aZ^2 \\equiv 0 \\ (mod \\ b)` so that `x^2 + |a|z^2` is minimal.

    Details
    =======

    Here ``w`` is a solution of the congruence `x^2 \\equiv a \\ (mod \\ b)`

    References
    ==========

    .. [1] Gaussian lattice Reduction [online]. Available:
           https://web.archive.org/web/20201021115213/http://home.ie.cuhk.edu.hk/~wkshum/wordpress/?p=404
    .. [2] Efficient Solution of Rational Conices, J. E. Cremona and D. Rusin,
           Mathematics of Computation, Volume 00, Number 0.
    """
    u = (0, 1)
    v = (1, 0)
    if dot(u, v, w, a, b) < 0:
        v = (-v[0], -v[1])
    if norm(u, w, a, b) < norm(v, w, a, b):
        u, v = (v, u)
    while norm(u, w, a, b) > norm(v, w, a, b):
        k = dot(u, v, w, a, b) // dot(v, v, w, a, b)
        u, v = (v, (u[0] - k * v[0], u[1] - k * v[1]))
    u, v = (v, u)
    if dot(u, v, w, a, b) < dot(v, v, w, a, b) / 2 or norm((u[0] - v[0], u[1] - v[1]), w, a, b) > norm(v, w, a, b):
        c = v
    else:
        c = (u[0] - v[0], u[1] - v[1])
    return (c[0] * w + b * c[1], c[0])
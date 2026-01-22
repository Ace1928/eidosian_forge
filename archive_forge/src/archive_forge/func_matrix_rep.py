from collections import defaultdict
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import I
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.dense import zeros
from sympy.printing.str import StrPrinter
from sympy.utilities.iterables import has_dups
def matrix_rep(op, basis):
    """
    Find the representation of an operator in a basis.

    Examples
    ========

    >>> from sympy.physics.secondquant import VarBosonicBasis, B, matrix_rep
    >>> b = VarBosonicBasis(5)
    >>> o = B(0)
    >>> matrix_rep(o, b)
    Matrix([
    [0, 1,       0,       0, 0],
    [0, 0, sqrt(2),       0, 0],
    [0, 0,       0, sqrt(3), 0],
    [0, 0,       0,       0, 2],
    [0, 0,       0,       0, 0]])
    """
    a = zeros(len(basis))
    for i in range(len(basis)):
        for j in range(len(basis)):
            a[i, j] = apply_operators(Dagger(basis[i]) * op * basis[j])
    return a
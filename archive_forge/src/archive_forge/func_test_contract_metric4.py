from sympy.concrete.summations import Sum
from sympy.core.function import expand
from sympy.core.numbers import Integer
from sympy.matrices.dense import (Matrix, eye)
from sympy.tensor.indexed import Indexed
from sympy.combinatorics import Permutation
from sympy.core import S, Rational, Symbol, Basic, Add
from sympy.core.containers import Tuple
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.tensor.array import Array
from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorSymmetry, \
from sympy.testing.pytest import raises, XFAIL, warns_deprecated_sympy
from sympy.matrices import diag
def test_contract_metric4():
    R3 = TensorIndexType('R3', dim=3)
    p, q, r = tensor_indices('p q r', R3)
    delta = R3.delta
    eps = R3.epsilon
    K = TensorHead('K', [R3])
    expr = eps(p, q, r) * (K(-p) * K(-q) + delta(-p, -q))
    assert expr.contract_metric(delta) == 0
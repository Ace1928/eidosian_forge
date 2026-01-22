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
def test_valued_tensor_expressions():
    with warns_deprecated_sympy():
        A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1, n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4 = _get_valued_base_test_variables()
        x1, x2, x3 = symbols('x1:4')
        rank2coeff = x1 * A(i3) * B(i2)
        assert rank2coeff[1, 1] == x1 * px
        assert rank2coeff[3, 3] == 3 * pz * x1
        coeff_expr = (x1 * A(i4) * (B(-i4) / x2)).data
        assert coeff_expr.expand() == -px * x1 / x2 - 2 * py * x1 / x2 - 3 * pz * x1 / x2
        add_expr = A(i0) + B(i0)
        assert add_expr[0] == E
        assert add_expr[1] == px + 1
        assert add_expr[2] == py + 2
        assert add_expr[3] == pz + 3
        sub_expr = A(i0) - B(i0)
        assert sub_expr[0] == E
        assert sub_expr[1] == px - 1
        assert sub_expr[2] == py - 2
        assert sub_expr[3] == pz - 3
        assert (add_expr * B(-i0)).data == -px - 2 * py - 3 * pz - 14
        expr1 = x1 * A(i0) + x2 * B(i0)
        expr2 = expr1 * B(i1) * -4
        expr3 = expr2 + 3 * x3 * AB(i0, i1)
        expr4 = expr3 / 2
        assert expr4 * 2 == expr3
        expr5 = expr4 * BA(-i1, -i0)
        assert expr5.data.expand() == 28 * E * x1 + 12 * px * x1 + 20 * py * x1 + 28 * pz * x1 + 136 * x2 + 3 * x3
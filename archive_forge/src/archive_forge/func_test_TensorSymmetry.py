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
def test_TensorSymmetry():
    assert TensorSymmetry.fully_symmetric(2) == TensorSymmetry(get_symmetric_group_sgs(2))
    assert TensorSymmetry.fully_symmetric(-3) == TensorSymmetry(get_symmetric_group_sgs(3, True))
    assert TensorSymmetry.direct_product(-4) == TensorSymmetry.fully_symmetric(-4)
    assert TensorSymmetry.fully_symmetric(-1) == TensorSymmetry.fully_symmetric(1)
    assert TensorSymmetry.direct_product(1, -1, 1) == TensorSymmetry.no_symmetry(3)
    assert TensorSymmetry(get_symmetric_group_sgs(2)) == TensorSymmetry(*get_symmetric_group_sgs(2))
    sym = TensorSymmetry.fully_symmetric(-3)
    assert sym.rank == 3
    assert sym.base == Tuple(0, 1)
    assert sym.generators == Tuple(Permutation(0, 1)(3, 4), Permutation(1, 2)(3, 4))
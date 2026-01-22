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
def test_TensorHead():
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    A = TensorHead('A', [Lorentz] * 2)
    assert A.name == 'A'
    assert A.index_types == [Lorentz, Lorentz]
    assert A.rank == 2
    assert A.symmetry == TensorSymmetry.no_symmetry(2)
    assert A.comm == 0
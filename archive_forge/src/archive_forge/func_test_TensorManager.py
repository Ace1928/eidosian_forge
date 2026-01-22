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
def test_TensorManager():
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    LorentzH = TensorIndexType('LorentzH', dummy_name='LH')
    i, j = tensor_indices('i,j', Lorentz)
    ih, jh = tensor_indices('ih,jh', LorentzH)
    p, q = tensor_heads('p q', [Lorentz])
    ph, qh = tensor_heads('ph qh', [LorentzH])
    Gsymbol = Symbol('Gsymbol')
    GHsymbol = Symbol('GHsymbol')
    TensorManager.set_comm(Gsymbol, GHsymbol, 0)
    G = TensorHead('G', [Lorentz], TensorSymmetry.no_symmetry(1), Gsymbol)
    assert TensorManager._comm_i2symbol[G.comm] == Gsymbol
    GH = TensorHead('GH', [LorentzH], TensorSymmetry.no_symmetry(1), GHsymbol)
    ps = G(i) * p(-i)
    psh = GH(ih) * ph(-ih)
    t = ps + psh
    t1 = t * t
    assert canon_bp(t1 - ps * ps - 2 * ps * psh - psh * psh) == 0
    qs = G(i) * q(-i)
    qsh = GH(ih) * qh(-ih)
    assert _is_equal(ps * qsh, qsh * ps)
    assert not _is_equal(ps * qs, qs * ps)
    n = TensorManager.comm_symbols2i(Gsymbol)
    assert TensorManager.comm_i2symbol(n) == Gsymbol
    assert GHsymbol in TensorManager._comm_symbols2i
    raises(ValueError, lambda: TensorManager.set_comm(GHsymbol, 1, 2))
    TensorManager.set_comms((Gsymbol, GHsymbol, 0), (Gsymbol, 1, 1))
    assert TensorManager.get_comm(n, 1) == TensorManager.get_comm(1, n) == 1
    TensorManager.clear()
    assert TensorManager.comm == [{0: 0, 1: 0, 2: 0}, {0: 0, 1: 1, 2: None}, {0: 0, 1: None}]
    assert GHsymbol not in TensorManager._comm_symbols2i
    nh = TensorManager.comm_symbols2i(GHsymbol)
    assert TensorManager.comm_i2symbol(nh) == GHsymbol
    assert GHsymbol in TensorManager._comm_symbols2i
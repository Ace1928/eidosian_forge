from sympy.matrices.dense import eye, Matrix
from sympy.tensor.tensor import tensor_indices, TensorHead, tensor_heads, \
from sympy.physics.hep.gamma_matrices import GammaMatrix as G, LorentzIndex, \
from sympy import Symbol
def test_bug_13636():
    """Test issue 13636 regarding handling traces of sums of products
    of GammaMatrix mixed with other factors."""
    pi, ki, pf = tensor_heads('pi, ki, pf', [LorentzIndex])
    i0, i1, i2, i3, i4 = tensor_indices('i0:5', LorentzIndex)
    x = Symbol('x')
    pis = pi(i2) * G(-i2)
    kis = ki(i3) * G(-i3)
    pfs = pf(i4) * G(-i4)
    a = pfs * G(i0) * kis * G(i1) * pis * G(-i1) * kis * G(-i0)
    b = pfs * G(i0) * kis * G(i1) * pis * x * G(-i0) * pi(-i1)
    ta = gamma_trace(a)
    tb = gamma_trace(b)
    t_a_plus_b = gamma_trace(a + b)
    assert ta == 4 * (-4 * ki(i0) * ki(-i0) * pf(i1) * pi(-i1) + 8 * ki(i0) * ki(i1) * pf(-i0) * pi(-i1))
    assert tb == -8 * x * ki(i0) * pf(-i0) * pi(i1) * pi(-i1)
    assert t_a_plus_b == ta + tb
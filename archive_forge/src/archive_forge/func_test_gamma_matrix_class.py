from sympy.matrices.dense import eye, Matrix
from sympy.tensor.tensor import tensor_indices, TensorHead, tensor_heads, \
from sympy.physics.hep.gamma_matrices import GammaMatrix as G, LorentzIndex, \
from sympy import Symbol
def test_gamma_matrix_class():
    i, j, k = tensor_indices('i,j,k', LorentzIndex)
    A = TensorHead('A', [LorentzIndex])
    t = A(k) * G(i) * G(-i)
    ts = simplify_gamma_expression(t)
    assert _is_tensor_eq(ts, Matrix([[4, 0, 0, 0], [0, 4, 0, 0], [0, 0, 4, 0], [0, 0, 0, 4]]) * A(k))
    t = G(i) * A(k) * G(j)
    ts = simplify_gamma_expression(t)
    assert _is_tensor_eq(ts, A(k) * G(i) * G(j))
    execute_gamma_simplify_tests_for_function(simplify_gamma_expression, D=4)
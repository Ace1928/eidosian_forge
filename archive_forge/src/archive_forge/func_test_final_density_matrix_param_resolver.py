import collections
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_final_density_matrix_param_resolver():
    s = sympy.Symbol('s')
    with pytest.raises(ValueError, match='not specified in parameter sweep'):
        _ = cirq.final_density_matrix(cirq.X ** s)
    np.testing.assert_allclose(cirq.final_density_matrix(cirq.X ** s, param_resolver={s: 0.5}), [[0.5 - 0j, 0.0 + 0.5j], [0.0 - 0.5j, 0.5 - 0j]])
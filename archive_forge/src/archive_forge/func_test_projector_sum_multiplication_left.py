import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
def test_projector_sum_multiplication_left():
    q0 = cirq.NamedQubit('q0')
    zero_projector_sum = cirq.ProjectorSum.from_projector_strings(cirq.ProjectorString({q0: 0}))
    multiplication_float = 2.0 * zero_projector_sum
    np.testing.assert_allclose(multiplication_float.matrix().toarray(), [[2.0, 0.0], [0.0, 0.0]])
    multiplication_int = 2 * zero_projector_sum
    np.testing.assert_allclose(multiplication_int.matrix().toarray(), [[2.0, 0.0], [0.0, 0.0]])
    multiplication_complex = 2j * zero_projector_sum
    np.testing.assert_allclose(multiplication_complex.matrix().toarray(), [[2j, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(zero_projector_sum.matrix().toarray(), [[1.0, 0.0], [0.0, 0.0]])
    with pytest.raises(TypeError):
        _ = 'not_the_correct_type' * zero_projector_sum
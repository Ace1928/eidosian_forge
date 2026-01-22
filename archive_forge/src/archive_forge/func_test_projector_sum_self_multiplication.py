import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
def test_projector_sum_self_multiplication():
    q0 = cirq.NamedQubit('q0')
    zero_projector_sum = cirq.ProjectorSum.from_projector_strings(cirq.ProjectorString({q0: 0}))
    multiplication_float = zero_projector_sum.copy()
    multiplication_float *= 2.0
    np.testing.assert_allclose(multiplication_float.matrix().toarray(), [[2.0, 0.0], [0.0, 0.0]])
    multiplication_int = zero_projector_sum.copy()
    multiplication_int *= 2
    np.testing.assert_allclose(multiplication_int.matrix().toarray(), [[2.0, 0.0], [0.0, 0.0]])
    multiplication_complex = zero_projector_sum.copy()
    multiplication_complex *= 2j
    np.testing.assert_allclose(multiplication_complex.matrix().toarray(), [[2j, 0.0], [0.0, 0.0]])
    with pytest.raises(TypeError):
        zero_projector_sum *= 'not_the_correct_type'
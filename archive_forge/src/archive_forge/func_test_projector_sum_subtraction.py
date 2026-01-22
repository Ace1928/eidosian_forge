import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
def test_projector_sum_subtraction():
    q0 = cirq.NamedQubit('q0')
    zero_projector_sum = cirq.ProjectorSum.from_projector_strings(cirq.ProjectorString({q0: 0}))
    one_projector_string = cirq.ProjectorString({q0: 1})
    one_projector_sum = cirq.ProjectorSum.from_projector_strings(one_projector_string)
    simple_subtraction = zero_projector_sum - one_projector_sum
    np.testing.assert_allclose(simple_subtraction.matrix().toarray(), [[1.0, 0.0], [0.0, -1.0]])
    simple_subtraction = zero_projector_sum - one_projector_string
    np.testing.assert_allclose(simple_subtraction.matrix().toarray(), [[1.0, 0.0], [0.0, -1.0]])
    np.testing.assert_allclose(zero_projector_sum.matrix().toarray(), [[1.0, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(one_projector_sum.matrix().toarray(), [[0.0, 0.0], [0.0, 1.0]])
    with pytest.raises(TypeError):
        _ = zero_projector_sum - 0.87539319
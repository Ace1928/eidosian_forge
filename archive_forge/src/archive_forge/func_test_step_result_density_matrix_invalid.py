import itertools
from typing import Optional
import pytest
import numpy as np
import cirq
import cirq.testing
from cirq import linalg
def test_step_result_density_matrix_invalid():
    q0, q1 = cirq.LineQubit.range(2)
    step_result = BasicStateVector({q0: 0})
    with pytest.raises(KeyError):
        step_result.density_matrix_of([q1])
    with pytest.raises(KeyError):
        step_result.density_matrix_of('junk')
    with pytest.raises(TypeError):
        step_result.density_matrix_of(0)
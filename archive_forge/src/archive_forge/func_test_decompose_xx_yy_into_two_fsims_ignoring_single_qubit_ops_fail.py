import itertools
import random
from typing import Any
import numpy as np
import pytest
import sympy
import cirq
from cirq.transformers.analytical_decompositions.two_qubit_to_fsim import (
def test_decompose_xx_yy_into_two_fsims_ignoring_single_qubit_ops_fail():
    c = _decompose_xx_yy_into_two_fsims_ignoring_single_qubit_ops(qubits=cirq.LineQubit.range(2), fsim_gate=cirq.FSimGate(theta=np.pi / 2, phi=0), canonical_x_kak_coefficient=np.pi / 4, canonical_y_kak_coefficient=np.pi / 8)
    np.testing.assert_allclose(cirq.kak_decomposition(cirq.Circuit(c)).interaction_coefficients, [np.pi / 4, np.pi / 8, 0])
    with pytest.raises(ValueError, match='Failed to synthesize'):
        _ = _decompose_xx_yy_into_two_fsims_ignoring_single_qubit_ops(qubits=cirq.LineQubit.range(2), fsim_gate=cirq.FSimGate(theta=np.pi / 5, phi=0), canonical_x_kak_coefficient=np.pi / 4, canonical_y_kak_coefficient=np.pi / 8)
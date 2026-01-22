import math
import cirq
import numpy
import pytest
import cirq_ionq as ionq
@pytest.mark.parametrize('phase', [0, 0.1, 0.4, math.pi / 2, math.pi, 2 * math.pi])
def test_gpi2_unitary(phase):
    """Tests that the GPI2 gate is unitary."""
    gate = ionq.GPI2Gate(phi=phase)
    mat = cirq.protocols.unitary(gate)
    numpy.testing.assert_array_almost_equal(mat.dot(mat.conj().T), numpy.identity(2))
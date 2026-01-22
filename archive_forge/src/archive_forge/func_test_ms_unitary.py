import math
import cirq
import numpy
import pytest
import cirq_ionq as ionq
@pytest.mark.parametrize('phases', [(0, 1), (0.1, 1), (0.4, 1), (math.pi / 2, 0), (0, math.pi), (0.1, 2 * math.pi)])
def test_ms_unitary(phases):
    """Tests that the MS gate is unitary."""
    gate = ionq.MSGate(phi0=phases[0], phi1=phases[1])
    mat = cirq.protocols.unitary(gate)
    numpy.testing.assert_array_almost_equal(mat.dot(mat.conj().T), numpy.identity(4))
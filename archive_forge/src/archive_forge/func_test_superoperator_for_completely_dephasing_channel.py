from typing import Iterable, Sequence
import numpy as np
import pytest
import cirq
def test_superoperator_for_completely_dephasing_channel():
    """Checks cirq.operation_to_superoperator on the completely dephasing channel."""
    assert np.all(cirq.operation_to_superoperator(cirq.phase_damp(1)) == np.diag([1, 0, 0, 1]))
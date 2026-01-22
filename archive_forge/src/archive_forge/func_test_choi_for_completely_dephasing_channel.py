from typing import Iterable, Sequence
import numpy as np
import pytest
import cirq
def test_choi_for_completely_dephasing_channel():
    """Checks cirq.operation_to_choi on the completely dephasing channel."""
    assert np.all(cirq.operation_to_choi(cirq.phase_damp(1)) == np.diag([1, 0, 0, 1]))
from typing import Any, Sequence
import numpy as np
import pytest
import cirq
from cirq.sim import simulation_state
from cirq.testing import PhaseUsingCleanAncilla, PhaseUsingDirtyAncilla
def test_rename_bad_dimensions():
    q0 = cirq.LineQubit(0)
    q1 = cirq.LineQid(1, 3)
    args = ExampleSimulationState()
    with pytest.raises(ValueError, match='Cannot rename to different dimensions'):
        args.rename(q0, q1)
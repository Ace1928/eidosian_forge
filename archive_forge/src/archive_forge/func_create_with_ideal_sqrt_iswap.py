from typing import (
import numpy as np
import cirq
from cirq import value
from cirq_google.calibration.phased_fsim import (
@classmethod
def create_with_ideal_sqrt_iswap(cls, *, simulator: Optional[cirq.Simulator]=None) -> 'PhasedFSimEngineSimulator':
    """Creates a PhasedFSimEngineSimulator that simulates ideal FSimGate(theta=π/4, phi=0).

        Attributes:
            simulator: Simulator object to use. When None, a new instance of cirq.Simulator() will
                be created.

        Returns:
            New PhasedFSimEngineSimulator instance.
        """

    def sample_gate(_1: cirq.Qid, _2: cirq.Qid, gate: cirq.FSimGate) -> PhasedFSimCharacterization:
        _assert_inv_sqrt_iswap_like(gate)
        return PhasedFSimCharacterization(theta=np.pi / 4, zeta=0.0, chi=0.0, gamma=0.0, phi=0.0)
    if simulator is None:
        simulator = cirq.Simulator()
    return cls(simulator, drift_generator=sample_gate, gates_translator=try_convert_sqrt_iswap_to_fsim)
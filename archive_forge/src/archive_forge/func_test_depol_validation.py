from typing import Dict, List, Set, Tuple
import numpy as np
import cirq
import pytest
from cirq.devices.noise_properties import NoiseModelFromNoiseProperties
from cirq.devices.superconducting_qubits_noise_properties import (
from cirq.devices.noise_utils import OpIdentifier, PHYSICAL_GATE_TAG
def test_depol_validation():
    q0, q1, q2 = cirq.LineQubit.range(3)
    z_2q_props = ExampleNoiseProperties(gate_times_ns=DEFAULT_GATE_NS, t1_ns={q0: 1}, tphi_ns={q0: 1}, readout_errors={q0: [0.1, 0.2]}, gate_pauli_errors={OpIdentifier(cirq.ZPowGate, q0, q1): 0.1}, validate=False)
    with pytest.raises(ValueError, match='takes 1 qubit'):
        _ = z_2q_props._depolarizing_error
    cz_3q_props = ExampleNoiseProperties(gate_times_ns=DEFAULT_GATE_NS, t1_ns={q0: 1}, tphi_ns={q0: 1}, readout_errors={q0: [0.1, 0.2]}, gate_pauli_errors={OpIdentifier(cirq.CZPowGate, q0, q1, q2): 0.1}, validate=False)
    with pytest.raises(ValueError, match='takes 2 qubit'):
        _ = cz_3q_props._depolarizing_error
    toffoli_props = ExampleNoiseProperties(gate_times_ns=DEFAULT_GATE_NS, t1_ns={q0: 1}, tphi_ns={q0: 1}, readout_errors={q0: [0.1, 0.2]}, gate_pauli_errors={OpIdentifier(cirq.CCXPowGate, q0, q1, q2): 0.1}, validate=False)
    with pytest.raises(ValueError):
        _ = toffoli_props._depolarizing_error
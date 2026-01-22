from typing import Dict, List, Tuple
from cirq.ops.fsim_gate import PhasedFSimGate
import numpy as np
import pytest
import cirq, cirq_google
from cirq.devices.noise_utils import OpIdentifier, PHYSICAL_GATE_TAG
from cirq_google.devices.google_noise_properties import (
def sample_noise_properties(system_qubits: List[cirq.Qid], qubit_pairs: List[Tuple[cirq.Qid, cirq.Qid]]):
    return GoogleNoiseProperties(gate_times_ns=DEFAULT_GATE_NS, t1_ns={q: 100000.0 for q in system_qubits}, tphi_ns={q: 200000.0 for q in system_qubits}, readout_errors={q: [SINGLE_QUBIT_ERROR, TWO_QUBIT_ERROR] for q in system_qubits}, gate_pauli_errors={**{OpIdentifier(g, q): 0.001 for g in GoogleNoiseProperties.single_qubit_gates() for q in system_qubits}, **{OpIdentifier(g, q0, q1): 0.01 for g in GoogleNoiseProperties.symmetric_two_qubit_gates() for q0, q1 in qubit_pairs}}, fsim_errors={OpIdentifier(g, q0, q1): cirq.PhasedFSimGate(0.01, 0.03, 0.04, 0.05, 0.02) for g in GoogleNoiseProperties.symmetric_two_qubit_gates() for q0, q1 in qubit_pairs})
from typing import Dict, List, Tuple
from cirq.ops.fsim_gate import PhasedFSimGate
import numpy as np
import pytest
import cirq, cirq_google
from cirq.devices.noise_utils import OpIdentifier, PHYSICAL_GATE_TAG
from cirq_google.devices.google_noise_properties import (
def test_with_params_fill():
    q0, q1 = cirq.LineQubit.range(2)
    props = sample_noise_properties([q0, q1], [(q0, q1), (q1, q0)])
    expected_vals = {'gate_times_ns': 91, 't1_ns': 92, 'tphi_ns': 93, 'readout_errors': [0.094, 0.095], 'gate_pauli_errors': 0.096, 'fsim_errors': cirq.PhasedFSimGate(0.0971, 0.0972, 0.0973, 0.0974, 0.0975)}
    props_v2 = props.with_params(gate_times_ns=expected_vals['gate_times_ns'], t1_ns=expected_vals['t1_ns'], tphi_ns=expected_vals['tphi_ns'], readout_errors=expected_vals['readout_errors'], gate_pauli_errors=expected_vals['gate_pauli_errors'], fsim_errors=expected_vals['fsim_errors'])
    assert props_v2 != props
    for key in props.gate_times_ns:
        assert key in props_v2.gate_times_ns
        assert props_v2.gate_times_ns[key] == expected_vals['gate_times_ns']
    for key in props.t1_ns:
        assert key in props_v2.t1_ns
        assert props_v2.t1_ns[key] == expected_vals['t1_ns']
    for key in props.tphi_ns:
        assert key in props_v2.tphi_ns
        assert props_v2.tphi_ns[key] == expected_vals['tphi_ns']
    for key in props.readout_errors:
        assert key in props_v2.readout_errors
        assert np.allclose(props_v2.readout_errors[key], expected_vals['readout_errors'])
    for key in props.gate_pauli_errors:
        assert key in props_v2.gate_pauli_errors
        assert props_v2.gate_pauli_errors[key] == expected_vals['gate_pauli_errors']
    for key in props.fsim_errors:
        assert key in props_v2.fsim_errors
        assert props_v2.fsim_errors[key] == expected_vals['fsim_errors']
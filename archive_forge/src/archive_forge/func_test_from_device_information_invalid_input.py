from dataclasses import dataclass
from typing import Dict, List, Tuple
import unittest.mock as mock
import pytest
import cirq
import cirq_google
from cirq_google.api import v2
from cirq_google.devices import grid_device
@pytest.mark.parametrize('error_match, qubit_pairs, gateset, gate_durations', [('Self loop encountered in qubit', [(cirq.GridQubit(0, 0), cirq.GridQubit(0, 0))], cirq.Gateset(), None), ('Unrecognized gate', [(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1))], cirq.Gateset(cirq.H), None), ('Some gate_durations keys are not found in gateset', [(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1))], cirq.Gateset(cirq.CZ), {cirq.GateFamily(cirq.SQRT_ISWAP): cirq.Duration(picos=1000)}), ('Multiple gate families .* expected to have the same duration value', [(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1))], cirq.Gateset(cirq.PhasedXZGate, cirq.XPowGate), {cirq.GateFamily(cirq.PhasedXZGate): cirq.Duration(picos=1000), cirq.GateFamily(cirq.XPowGate): cirq.Duration(picos=2000)})])
def test_from_device_information_invalid_input(error_match, qubit_pairs, gateset, gate_durations):
    with pytest.raises(ValueError, match=error_match):
        grid_device.GridDevice._from_device_information(qubit_pairs=qubit_pairs, gateset=gateset, gate_durations=gate_durations)
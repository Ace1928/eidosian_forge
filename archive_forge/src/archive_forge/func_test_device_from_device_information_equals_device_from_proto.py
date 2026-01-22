from dataclasses import dataclass
from typing import Dict, List, Tuple
import unittest.mock as mock
import pytest
import cirq
import cirq_google
from cirq_google.api import v2
from cirq_google.devices import grid_device
def test_device_from_device_information_equals_device_from_proto():
    device_info, spec = _create_device_spec_with_isolated_qubits()
    gateset = cirq.Gateset(cirq_google.SYC, cirq.SQRT_ISWAP, cirq.SQRT_ISWAP_INV, cirq.CZ, cirq.ops.phased_x_z_gate.PhasedXZGate, cirq.GateFamily(cirq.ops.common_gates.ZPowGate, tags_to_ignore=[cirq_google.PhysicalZTag()]), cirq.GateFamily(cirq.ops.common_gates.ZPowGate, tags_to_accept=[cirq_google.PhysicalZTag()]), cirq_google.experimental.ops.coupler_pulse.CouplerPulse, cirq.ops.measurement_gate.MeasurementGate, cirq.ops.wait_gate.WaitGate)
    base_duration = cirq.Duration(picos=1000)
    gate_durations = {cirq.GateFamily(cirq_google.SYC): base_duration * 0, cirq.GateFamily(cirq.SQRT_ISWAP): base_duration * 1, cirq.GateFamily(cirq.SQRT_ISWAP_INV): base_duration * 2, cirq.GateFamily(cirq.CZ): base_duration * 3, cirq.GateFamily(cirq.ops.phased_x_z_gate.PhasedXZGate): base_duration * 4, cirq.GateFamily(cirq.ops.common_gates.ZPowGate, tags_to_ignore=[cirq_google.PhysicalZTag()]): base_duration * 5, cirq.GateFamily(cirq.ops.common_gates.ZPowGate, tags_to_accept=[cirq_google.PhysicalZTag()]): base_duration * 6, cirq.GateFamily(cirq_google.experimental.ops.coupler_pulse.CouplerPulse): base_duration * 7, cirq.GateFamily(cirq.ops.measurement_gate.MeasurementGate): base_duration * 8, cirq.GateFamily(cirq.ops.wait_gate.WaitGate): base_duration * 9}
    device_from_information = cirq_google.GridDevice._from_device_information(qubit_pairs=device_info.qubit_pairs, gateset=gateset, gate_durations=gate_durations, all_qubits=device_info.grid_qubits)
    assert device_from_information == cirq_google.GridDevice.from_proto(spec)
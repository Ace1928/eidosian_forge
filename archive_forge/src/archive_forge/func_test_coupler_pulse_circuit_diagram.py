import pytest
import sympy
import cirq
import cirq_google.experimental.ops.coupler_pulse as coupler_pulse
def test_coupler_pulse_circuit_diagram():
    a, b = cirq.LineQubit.range(2)
    gate = coupler_pulse.CouplerPulse(hold_time=cirq.Duration(nanos=10), coupling_mhz=25.0, rise_time=cirq.Duration(nanos=18))
    circuit = cirq.Circuit(gate(a, b))
    cirq.testing.assert_has_diagram(circuit, '\n0: ───/‾‾(10 ns@25.0MHz)‾‾\\───\n      │\n1: ───/‾‾(10 ns@25.0MHz)‾‾\\───\n')
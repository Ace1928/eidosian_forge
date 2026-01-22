import pytest
import sympy
import cirq
def test_periodic_value_approx_eq_boundary():
    assert cirq.approx_eq(cirq.PeriodicValue(0.0, 2.0), cirq.PeriodicValue(1.9, 2.0), atol=0.2)
    assert cirq.approx_eq(cirq.PeriodicValue(0.1, 2.0), cirq.PeriodicValue(1.9, 2.0), atol=0.3)
    assert cirq.approx_eq(cirq.PeriodicValue(1.9, 2.0), cirq.PeriodicValue(0.1, 2.0), atol=0.3)
    assert not cirq.approx_eq(cirq.PeriodicValue(0.1, 2.0), cirq.PeriodicValue(1.9, 2.0), atol=0.1)
    assert cirq.approx_eq(cirq.PeriodicValue(0, 1.0), cirq.PeriodicValue(0.5, 1.0), atol=0.6)
    assert not cirq.approx_eq(cirq.PeriodicValue(0, 1.0), cirq.PeriodicValue(0.5, 1.0), atol=0.1)
    assert cirq.approx_eq(cirq.PeriodicValue(0.4, 1.0), cirq.PeriodicValue(0.6, 1.0), atol=0.3)
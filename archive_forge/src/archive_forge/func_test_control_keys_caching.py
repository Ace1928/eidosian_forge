import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_control_keys_caching():
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    m = cirq.Moment(cirq.X(q0).with_classical_controls('foo'))
    assert m._control_keys is None
    keys = cirq.control_keys(m)
    assert m._control_keys == keys
    m = m.with_operation(cirq.X(q1).with_classical_controls('bar'))
    assert m._control_keys == {cirq.MeasurementKey(name='bar'), cirq.MeasurementKey(name='foo')}
    m = m.with_operations(cirq.X(q2).with_classical_controls('doh'), cirq.X(q3).with_classical_controls('baz'))
    assert m._control_keys == {cirq.MeasurementKey(name='bar'), cirq.MeasurementKey(name='foo'), cirq.MeasurementKey(name='doh'), cirq.MeasurementKey(name='baz')}
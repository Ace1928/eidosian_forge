import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
from cirq.transformers.measurement_transformers import _ConfusionChannel, _MeasurementQid, _mod_add
def test_multiple_sympy_control_complex():
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.measure(q0, key='a'), cirq.measure(q1, key='b'), cirq.X(q2).with_classical_controls(sympy_parser.parse_expr('a >= b')).with_classical_controls(sympy_parser.parse_expr('a <= b')), cirq.measure(q2, key='c'))
    assert_equivalent_to_deferred(circuit)
    deferred = cirq.defer_measurements(circuit)
    q_ma = _MeasurementQid('a', q0)
    q_mb = _MeasurementQid('b', q1)
    cirq.testing.assert_same_circuits(deferred, cirq.Circuit(cirq.CX(q0, q_ma), cirq.CX(q1, q_mb), cirq.ControlledOperation([q_ma, q_mb], cirq.X(q2), cirq.SumOfProducts([[0, 0], [1, 1]])), cirq.measure(q_ma, key='a'), cirq.measure(q_mb, key='b'), cirq.measure(q2, key='c')))
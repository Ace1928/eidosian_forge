import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
from cirq.transformers.measurement_transformers import _ConfusionChannel, _MeasurementQid, _mod_add
def test_sympy_control_complex_qudit():
    q0, q1, q2 = cirq.LineQid.for_qid_shape((4, 2, 2))
    circuit = cirq.Circuit(cirq.measure(q0, key='a'), cirq.measure(q1, key='b'), cirq.X(q2).with_classical_controls(sympy_parser.parse_expr('a > b')), cirq.measure(q2, key='c'))
    assert_equivalent_to_deferred(circuit)
    deferred = cirq.defer_measurements(circuit)
    q_ma = _MeasurementQid('a', q0)
    q_mb = _MeasurementQid('b', q1)
    cirq.testing.assert_same_circuits(deferred, cirq.Circuit(_mod_add(q0, q_ma), cirq.CX(q1, q_mb), cirq.ControlledOperation([q_ma, q_mb], cirq.X(q2), cirq.SumOfProducts([[1, 0], [2, 0], [3, 0], [2, 1], [3, 1]])), cirq.measure(q_ma, key='a'), cirq.measure(q_mb, key='b'), cirq.measure(q2, key='c')))
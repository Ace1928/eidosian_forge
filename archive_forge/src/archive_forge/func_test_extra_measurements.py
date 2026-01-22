import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
from cirq.transformers.measurement_transformers import _ConfusionChannel, _MeasurementQid, _mod_add
def test_extra_measurements():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.measure(q0, key='a'), cirq.measure(q0, key='b'), cirq.X(q1).with_classical_controls('a'), cirq.measure(q1, key='c'))
    assert_equivalent_to_deferred(circuit)
    deferred = cirq.defer_measurements(circuit)
    q_ma = _MeasurementQid('a', q0)
    cirq.testing.assert_same_circuits(deferred, cirq.Circuit(cirq.CX(q0, q_ma), cirq.CX(q_ma, q1), cirq.measure(q_ma, key='a'), cirq.measure(q0, key='b'), cirq.measure(q1, key='c')))
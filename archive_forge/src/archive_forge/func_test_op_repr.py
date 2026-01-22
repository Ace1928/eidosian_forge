from typing import cast
import numpy as np
import pytest
import cirq
def test_op_repr():
    a, b = cirq.LineQubit.range(2)
    assert repr(cirq.measure(a)) == 'cirq.measure(cirq.LineQubit(0))'
    assert repr(cirq.measure(a, b)) == 'cirq.measure(cirq.LineQubit(0), cirq.LineQubit(1))'
    assert repr(cirq.measure(a, b, key='out', invert_mask=(False, True))) == "cirq.measure(cirq.LineQubit(0), cirq.LineQubit(1), key=cirq.MeasurementKey(name='out'), invert_mask=(False, True))"
    assert repr(cirq.measure(a, b, key='out', invert_mask=(False, True), confusion_map={(0,): np.array([[0, 1], [1, 0]], dtype=np.dtype('int64'))})) == "cirq.measure(cirq.LineQubit(0), cirq.LineQubit(1), key=cirq.MeasurementKey(name='out'), invert_mask=(False, True), confusion_map={(0,): np.array([[0, 1], [1, 0]], dtype=np.dtype('int64'))})"
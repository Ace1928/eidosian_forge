from typing import cast
import numpy as np
import pytest
import cirq
def test_measurement_eq():
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: cirq.MeasurementGate(1, 'a'), lambda: cirq.MeasurementGate(1, 'a', invert_mask=()), lambda: cirq.MeasurementGate(1, 'a', qid_shape=(2,)), lambda: cirq.MeasurementGate(1, 'a', confusion_map={}))
    eq.add_equality_group(cirq.MeasurementGate(1, 'a', invert_mask=(True,)))
    eq.add_equality_group(cirq.MeasurementGate(1, 'a', invert_mask=(False,)))
    eq.add_equality_group(cirq.MeasurementGate(1, 'a', confusion_map={(0,): np.array([[0, 1], [1, 0]])}))
    eq.add_equality_group(cirq.MeasurementGate(1, 'b'))
    eq.add_equality_group(cirq.MeasurementGate(2, 'a'))
    eq.add_equality_group(cirq.MeasurementGate(3, 'a'), cirq.MeasurementGate(3, 'a', qid_shape=(2, 2, 2)))
    eq.add_equality_group(cirq.MeasurementGate(3, 'a', qid_shape=(1, 2, 3)))
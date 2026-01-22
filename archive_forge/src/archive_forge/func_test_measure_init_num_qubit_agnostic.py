from typing import cast
import numpy as np
import pytest
import cirq
def test_measure_init_num_qubit_agnostic():
    assert cirq.qid_shape(cirq.MeasurementGate(3, 'a', qid_shape=(1, 2, 3))) == (1, 2, 3)
    assert cirq.qid_shape(cirq.MeasurementGate(key='a', qid_shape=(1, 2, 3))) == (1, 2, 3)
    with pytest.raises(ValueError, match='len.* >'):
        cirq.MeasurementGate(5, 'a', invert_mask=(True,) * 6)
    with pytest.raises(ValueError, match='len.* !='):
        cirq.MeasurementGate(5, 'a', qid_shape=(1, 2))
    with pytest.raises(ValueError, match='valid string'):
        cirq.MeasurementGate(2, qid_shape=(1, 2), key=None)
    with pytest.raises(ValueError, match='Confusion matrices have index out of bounds'):
        cirq.MeasurementGate(1, 'a', confusion_map={(1,): np.array([[0, 1], [1, 0]])})
    with pytest.raises(ValueError, match='Specify either'):
        cirq.MeasurementGate()
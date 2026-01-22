import numpy as np
import cirq
import pytest
from cirq.experiments.single_qubit_readout_calibration_test import NoisySingleQubitReadoutSampler
def test_from_measurement():
    qubits = cirq.LineQubit.range(3)
    confuse_02 = np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]])
    confuse_1 = np.array([[0, 1], [1, 0]])
    op = cirq.measure(*qubits, key='a', invert_mask=(True, False), confusion_map={(0, 2): confuse_02, (1,): confuse_1})
    tcm = cirq.TensoredConfusionMatrices.from_measurement(op.gate, op.qubits)
    expected_tcm = cirq.TensoredConfusionMatrices([confuse_02, confuse_1], ((qubits[0], qubits[2]), (qubits[1],)), repetitions=0, timestamp=0)
    assert tcm == expected_tcm
    no_cm_op = cirq.measure(*qubits, key='a')
    with pytest.raises(ValueError, match='Measurement has no confusion matrices'):
        _ = cirq.TensoredConfusionMatrices.from_measurement(no_cm_op.gate, no_cm_op.qubits)
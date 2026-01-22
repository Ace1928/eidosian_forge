from typing import cast
import numpy as np
import pytest
import cirq
def test_confused_measure_qasm():
    q0 = cirq.LineQubit(0)
    assert cirq.qasm(cirq.measure(q0, key='a', confusion_map={(0,): np.array([[0, 1], [1, 0]])}), args=cirq.QasmArgs(), default='not implemented') == 'not implemented'
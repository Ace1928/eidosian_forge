import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
from cirq.transformers.measurement_transformers import _ConfusionChannel, _MeasurementQid, _mod_add
def test_confusion_channel_consistency():
    two_d_chan = _ConfusionChannel(np.array([[0.5, 0.5], [0.4, 0.6]]), shape=(2,))
    cirq.testing.assert_has_consistent_apply_channel(two_d_chan)
    three_d_chan = _ConfusionChannel(np.array([[0.5, 0.3, 0.2], [0.4, 0.5, 0.1], [0, 0, 1]]), shape=(3,))
    cirq.testing.assert_has_consistent_apply_channel(three_d_chan)
    two_q_chan = _ConfusionChannel(np.array([[0.5, 0.3, 0.1, 0.1], [0.4, 0.5, 0.1, 0], [0, 0, 1, 0], [0, 0, 0.5, 0.5]]), shape=(2, 2))
    cirq.testing.assert_has_consistent_apply_channel(two_q_chan)
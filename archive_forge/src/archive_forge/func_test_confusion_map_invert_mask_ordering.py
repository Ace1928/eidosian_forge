import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
from cirq.transformers.measurement_transformers import _ConfusionChannel, _MeasurementQid, _mod_add
def test_confusion_map_invert_mask_ordering():
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.measure(q0, key='a', confusion_map={(0,): np.array([[1, 0], [1, 0]])}, invert_mask=(1,)), cirq.I(q0))
    assert_equivalent_to_deferred(circuit)
import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
def test_measure_at_end_invert_mask_multiple_qubits():
    simulator = cirq.Simulator()
    a, b, c = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.measure(a, key='a', invert_mask=(True,)), cirq.measure(b, c, key='bc', invert_mask=(False, True)))
    result = simulator.run(circuit, repetitions=4)
    np.testing.assert_equal(result.measurements['a'], np.array([[True]] * 4))
    np.testing.assert_equal(result.measurements['bc'], np.array([[0, 1]] * 4))
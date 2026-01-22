import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_simulate_with_invert_mask():
    q0, q1, q2, q3, q4 = cirq.LineQid.for_qid_shape((2, 3, 3, 3, 4))
    c = cirq.Circuit(cirq.XPowGate(dimension=2)(q0), cirq.XPowGate(dimension=3)(q2), cirq.XPowGate(dimension=3)(q3) ** 2, cirq.XPowGate(dimension=4)(q4) ** 3, cirq.measure(q0, q1, q2, q3, q4, key='a', invert_mask=(True,) * 4))
    assert np.all(cirq.DensityMatrixSimulator().run(c).measurements['a'] == [[0, 1, 0, 2, 3]])
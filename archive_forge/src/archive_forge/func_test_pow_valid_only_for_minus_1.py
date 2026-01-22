import itertools
import os
import time
from collections import defaultdict
from random import randint, random, sample, randrange
from typing import Iterator, Optional, Tuple, TYPE_CHECKING
import numpy as np
import pytest
import sympy
import cirq
from cirq import circuits
from cirq import ops
from cirq.testing.devices import ValidatingTestDevice
@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_pow_valid_only_for_minus_1(circuit_cls):
    a, b = cirq.LineQubit.range(2)
    forward = circuit_cls((cirq.X ** 0.5)(a), (cirq.Y ** (-0.2))(b), cirq.CZ(a, b))
    backward = circuit_cls((cirq.CZ ** (-1.0))(a, b), (cirq.X ** (-0.5))(a), (cirq.Y ** 0.2)(b))
    cirq.testing.assert_same_circuits(cirq.pow(forward, -1), backward)
    with pytest.raises(TypeError, match='__pow__'):
        cirq.pow(forward, 1)
    with pytest.raises(TypeError, match='__pow__'):
        cirq.pow(forward, 0)
    with pytest.raises(TypeError, match='__pow__'):
        cirq.pow(forward, -2.5)
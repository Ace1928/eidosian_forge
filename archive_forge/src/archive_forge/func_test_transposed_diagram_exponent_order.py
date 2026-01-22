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
def test_transposed_diagram_exponent_order(circuit_cls):
    a, b, c = cirq.LineQubit.range(3)
    circuit = circuit_cls(cirq.CZ(a, b) ** (-0.5), cirq.CZ(a, c) ** 0.5, cirq.CZ(b, c) ** 0.125)
    cirq.testing.assert_has_diagram(circuit, '\n0 1      2\n│ │      │\n@─@^-0.5 │\n│ │      │\n@─┼──────@^0.5\n│ │      │\n│ @──────@^(1/8)\n│ │      │\n', transpose=True)
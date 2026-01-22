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
def test_circuit_diagram_on_gate_without_info(circuit_cls):
    q = cirq.NamedQubit('(0, 0)')
    q2 = cirq.NamedQubit('(0, 1)')
    q3 = cirq.NamedQubit('(0, 2)')

    class FGate(cirq.Gate):

        def __init__(self, num_qubits=1):
            self._num_qubits = num_qubits

        def num_qubits(self) -> int:
            return self._num_qubits

        def __repr__(self):
            return 'python-object-FGate:arbitrary-digits'
    f = FGate()
    cirq.testing.assert_has_diagram(circuit_cls([cirq.Moment([f.on(q)])]), '\n(0, 0): ---python-object-FGate:arbitrary-digits---\n', use_unicode_characters=False)
    f3 = FGate(3)
    cirq.testing.assert_has_diagram(circuit_cls([cirq.Moment([f3.on(q, q3, q2)])]), '\n(0, 0): ---python-object-FGate:arbitrary-digits---\n           |\n(0, 1): ---#3-------------------------------------\n           |\n(0, 2): ---#2-------------------------------------\n', use_unicode_characters=False)
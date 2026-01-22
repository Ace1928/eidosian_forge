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
def test_expanding_gate_symbols(circuit_cls):

    class MultiTargetCZ(cirq.Gate):

        def __init__(self, num_qubits):
            self._num_qubits = num_qubits

        def num_qubits(self) -> int:
            return self._num_qubits

        def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> Tuple[str, ...]:
            assert args.known_qubit_count is not None
            return ('@',) + ('Z',) * (args.known_qubit_count - 1)
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    t0 = circuit_cls(MultiTargetCZ(1).on(c))
    t1 = circuit_cls(MultiTargetCZ(2).on(c, a))
    t2 = circuit_cls(MultiTargetCZ(3).on(c, a, b))
    cirq.testing.assert_has_diagram(t0, '\nc: ───@───\n')
    cirq.testing.assert_has_diagram(t1, '\na: ───Z───\n      │\nc: ───@───\n')
    cirq.testing.assert_has_diagram(t2, '\na: ───Z───\n      │\nb: ───Z───\n      │\nc: ───@───\n')
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
def test_repr_html_escaping(circuit_cls):

    class TestGate(cirq.Gate):

        def num_qubits(self):
            return 2

        def _circuit_diagram_info_(self, args):
            return cirq.CircuitDiagramInfo(wire_symbols=["< ' F ' >", "< ' F ' >"])
    F2 = TestGate()
    a = cirq.LineQubit(1)
    c = cirq.NamedQubit('|c>')
    circuit = circuit_cls([F2(a, c)])
    assert '&lt; &#x27; F &#x27; &gt;' in circuit._repr_html_()
    assert '|c&gt;' in circuit._repr_html_()
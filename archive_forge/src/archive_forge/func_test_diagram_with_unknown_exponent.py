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
def test_diagram_with_unknown_exponent(circuit_cls):

    class WeirdGate(cirq.testing.SingleQubitGate):

        def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
            return cirq.CircuitDiagramInfo(wire_symbols=('B',), exponent='fancy')

    class WeirderGate(cirq.testing.SingleQubitGate):

        def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
            return cirq.CircuitDiagramInfo(wire_symbols=('W',), exponent='fancy-that')
    c = circuit_cls(WeirdGate().on(cirq.NamedQubit('q')), WeirderGate().on(cirq.NamedQubit('q')))
    cirq.testing.assert_has_diagram(c, 'q: ───B^fancy───W^(fancy-that)───')
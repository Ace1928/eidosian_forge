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
def test_text_diagram_jupyter(circuit_cls):
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    circuit = circuit_cls((cirq.CNOT(a, b), cirq.CNOT(b, c), cirq.CNOT(c, a)) * 50)
    text_expected = circuit.to_text_diagram()

    class FakePrinter:

        def __init__(self):
            self.text_pretty = ''

        def text(self, to_print):
            self.text_pretty += to_print
    p = FakePrinter()
    circuit._repr_pretty_(p, False)
    assert p.text_pretty == text_expected
    p = FakePrinter()
    circuit._repr_pretty_(p, True)
    assert p.text_pretty == f'{circuit_cls.__name__}(...)'
    text_html = circuit._repr_html_()
    assert text_expected in text_html
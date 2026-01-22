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
def test_to_text_diagram_multi_qubit_gate(circuit_cls):
    q1 = cirq.NamedQubit('(0, 0)')
    q2 = cirq.NamedQubit('(0, 1)')
    q3 = cirq.NamedQubit('(0, 2)')
    c = circuit_cls(cirq.measure(q1, q2, q3, key='msg'))
    cirq.testing.assert_has_diagram(c, "\n(0, 0): ───M('msg')───\n           │\n(0, 1): ───M──────────\n           │\n(0, 2): ───M──────────\n")
    cirq.testing.assert_has_diagram(c, "\n(0, 0): ---M('msg')---\n           |\n(0, 1): ---M----------\n           |\n(0, 2): ---M----------\n", use_unicode_characters=False)
    cirq.testing.assert_has_diagram(c, "\n(0, 0)   (0, 1) (0, 2)\n│        │      │\nM('msg')─M──────M\n│        │      │\n", transpose=True)
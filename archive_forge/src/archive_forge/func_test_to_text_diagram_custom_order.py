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
def test_to_text_diagram_custom_order(circuit_cls):
    qa = cirq.NamedQubit('2')
    qb = cirq.NamedQubit('3')
    qc = cirq.NamedQubit('4')
    c = circuit_cls([cirq.Moment([cirq.X(qa), cirq.X(qb), cirq.X(qc)])])
    cirq.testing.assert_has_diagram(c, '\n3: ---X---\n\n4: ---X---\n\n2: ---X---\n', qubit_order=cirq.QubitOrder.sorted_by(lambda e: int(str(e)) % 3), use_unicode_characters=False)
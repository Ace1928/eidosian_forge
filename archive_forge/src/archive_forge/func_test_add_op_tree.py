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
def test_add_op_tree(circuit_cls):
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = circuit_cls()
    assert c + [cirq.X(a), cirq.Y(b)] == circuit_cls([cirq.Moment([cirq.X(a), cirq.Y(b)])])
    assert c + cirq.X(a) == circuit_cls(cirq.X(a))
    assert c + [cirq.X(a)] == circuit_cls(cirq.X(a))
    assert c + [[[cirq.X(a)], []]] == circuit_cls(cirq.X(a))
    assert c + (cirq.X(a),) == circuit_cls(cirq.X(a))
    assert c + (cirq.X(a) for _ in range(1)) == circuit_cls(cirq.X(a))
    with pytest.raises(TypeError):
        _ = c + cirq.X
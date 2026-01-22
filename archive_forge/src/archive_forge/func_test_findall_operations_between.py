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
def test_findall_operations_between(circuit_cls):
    a, b, c, d = cirq.LineQubit.range(4)
    circuit = circuit_cls(cirq.H(a), cirq.CZ(a, b), cirq.H(b), cirq.CZ(b, c), cirq.H(c), cirq.CZ(c, d), cirq.H(d), cirq.CZ(c, d), cirq.H(c), cirq.CZ(b, c), cirq.H(b), cirq.CZ(a, b), cirq.H(a))
    actual = circuit.findall_operations_between(start_frontier={}, end_frontier={})
    assert actual == []
    actual = circuit.findall_operations_between(start_frontier={a: 5}, end_frontier={a: 5})
    assert actual == []
    actual = circuit.findall_operations_between(start_frontier={a: 5}, end_frontier={})
    assert actual == [(11, cirq.CZ(a, b)), (12, cirq.H(a))]
    actual = circuit.findall_operations_between(start_frontier={}, end_frontier={a: 5})
    assert actual == [(0, cirq.H(a)), (1, cirq.CZ(a, b))]
    actual = circuit.findall_operations_between(start_frontier={a: 5}, end_frontier={}, omit_crossing_operations=True)
    assert actual == [(12, cirq.H(a))]
    actual = circuit.findall_operations_between(start_frontier={a: 5, b: 5}, end_frontier={}, omit_crossing_operations=True)
    assert actual == [(10, cirq.H(b)), (11, cirq.CZ(a, b)), (12, cirq.H(a))]
    actual = circuit.findall_operations_between(start_frontier={a: 5}, end_frontier={b: 5})
    assert actual == [(1, cirq.CZ(a, b)), (2, cirq.H(b)), (3, cirq.CZ(b, c)), (11, cirq.CZ(a, b)), (12, cirq.H(a))]
    actual = circuit.findall_operations_between(start_frontier={a: 5}, end_frontier={a: 5, b: 5})
    assert actual == [(1, cirq.CZ(a, b)), (2, cirq.H(b)), (3, cirq.CZ(b, c))]
    actual = circuit.findall_operations_between(start_frontier={c: 4}, end_frontier={c: 8})
    assert actual == [(4, cirq.H(c)), (5, cirq.CZ(c, d)), (7, cirq.CZ(c, d))]
    actual = circuit.findall_operations_between(start_frontier={a: -100}, end_frontier={a: +100})
    assert actual == [(0, cirq.H(a)), (1, cirq.CZ(a, b)), (11, cirq.CZ(a, b)), (12, cirq.H(a))]
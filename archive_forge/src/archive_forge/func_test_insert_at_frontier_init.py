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
def test_insert_at_frontier_init():
    x = cirq.NamedQubit('x')
    op = cirq.X(x)
    circuit = cirq.Circuit(op)
    actual_frontier = circuit.insert_at_frontier(op, 3)
    expected_circuit = cirq.Circuit([cirq.Moment([op]), cirq.Moment(), cirq.Moment(), cirq.Moment([op])])
    assert circuit == expected_circuit
    expected_frontier = defaultdict(lambda: 0)
    expected_frontier[x] = 4
    assert actual_frontier == expected_frontier
    with pytest.raises(ValueError):
        circuit = cirq.Circuit([cirq.Moment(), cirq.Moment([op])])
        frontier = {x: 2}
        circuit.insert_at_frontier(op, 0, frontier)
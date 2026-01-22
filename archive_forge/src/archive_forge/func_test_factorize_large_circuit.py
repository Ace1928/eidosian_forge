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
def test_factorize_large_circuit():
    circuit = cirq.Circuit()
    qubits = cirq.GridQubit.rect(3, 3)
    circuit.append(cirq.Moment((cirq.X(q) for q in qubits)))
    pairset = [[(0, 2), (4, 6)], [(1, 2), (4, 8)]]
    for pairs in pairset:
        circuit.append(cirq.Moment((cirq.CZ(qubits[a], qubits[b]) for a, b in pairs)))
    circuit.append(cirq.Moment((cirq.Y(q) for q in qubits)))
    factors = list(circuit.factorize())
    desired = ['\n(0, 0): ───X───@───────Y───\n               │\n(0, 1): ───X───┼───@───Y───\n               │   │\n(0, 2): ───X───@───@───Y───\n', '\n(1, 0): ───X───────────Y───\n', '\n(1, 1): ───X───@───@───Y───\n               │   │\n(2, 0): ───X───@───┼───Y───\n                   │\n(2, 2): ───X───────@───Y───\n', '\n(1, 2): ───X───────────Y───\n', '\n(2, 1): ───X───────────Y───\n    ']
    assert len(factors) == 5
    for f, d in zip(factors, desired):
        cirq.testing.assert_has_diagram(f, d)
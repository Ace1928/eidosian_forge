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
def test_factorize_simple_circuit_two_factors():
    circuit = cirq.Circuit()
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit.append([cirq.H(q1), cirq.CZ(q0, q1), cirq.H(q2), cirq.H(q0), cirq.H(q0)])
    factors = list(circuit.factorize())
    assert len(factors) == 2
    desired = ['\n0: ───────@───H───H───\n          │\n1: ───H───@───────────\n', '\n2: ───H───────────────\n']
    for f, d in zip(factors, desired):
        cirq.testing.assert_has_diagram(f, d)
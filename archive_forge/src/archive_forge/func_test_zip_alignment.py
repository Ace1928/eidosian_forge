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
def test_zip_alignment(circuit_cls):
    a, b, c = cirq.LineQubit.range(3)
    circuit1 = circuit_cls([cirq.H(a)] * 5)
    circuit2 = circuit_cls([cirq.H(b)] * 3)
    circuit3 = circuit_cls([cirq.H(c)] * 2)
    c_start = circuit_cls.zip(circuit1, circuit2, circuit3, align='LEFT')
    assert c_start == circuit_cls(cirq.Moment(cirq.H(a), cirq.H(b), cirq.H(c)), cirq.Moment(cirq.H(a), cirq.H(b), cirq.H(c)), cirq.Moment(cirq.H(a), cirq.H(b)), cirq.Moment(cirq.H(a)), cirq.Moment(cirq.H(a)))
    c_end = circuit_cls.zip(circuit1, circuit2, circuit3, align='RIGHT')
    assert c_end == circuit_cls(cirq.Moment(cirq.H(a)), cirq.Moment(cirq.H(a)), cirq.Moment(cirq.H(a), cirq.H(b)), cirq.Moment(cirq.H(a), cirq.H(b), cirq.H(c)), cirq.Moment(cirq.H(a), cirq.H(b), cirq.H(c)))
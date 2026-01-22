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
def test_add_iadd_equivalence():
    q0, q1 = cirq.LineQubit.range(2)
    iadd_circuit = cirq.Circuit(cirq.X(q0))
    iadd_circuit += cirq.H(q1)
    add_circuit = cirq.Circuit(cirq.X(q0)) + cirq.H(q1)
    assert iadd_circuit == add_circuit
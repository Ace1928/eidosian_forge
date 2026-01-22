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
def test_insert_operations_errors():
    a, b, c = (cirq.NamedQubit(s) for s in 'abc')
    with pytest.raises(ValueError):
        circuit = cirq.Circuit([cirq.Moment([cirq.Z(c)])])
        operations = [cirq.X(a), cirq.CZ(a, b)]
        insertion_indices = [0, 0]
        circuit._insert_operations(operations, insertion_indices)
    with pytest.raises(ValueError):
        circuit = cirq.Circuit(cirq.X(a))
        operations = [cirq.CZ(a, b)]
        insertion_indices = [0]
        circuit._insert_operations(operations, insertion_indices)
    with pytest.raises(ValueError):
        circuit = cirq.Circuit()
        operations = [cirq.X(a), cirq.CZ(a, b)]
        insertion_indices = []
        circuit._insert_operations(operations, insertion_indices)
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
def test_indexing_by_numpy_integer(circuit_cls):
    q = cirq.NamedQubit('q')
    c = circuit_cls(cirq.X(q), cirq.Y(q))
    assert c[np.int32(1)] == cirq.Moment([cirq.Y(q)])
    assert c[np.int64(1)] == cirq.Moment([cirq.Y(q)])
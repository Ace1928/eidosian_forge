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
def test_insert_into_range():
    x = cirq.NamedQubit('x')
    y = cirq.NamedQubit('y')
    c = cirq.Circuit([cirq.Moment([cirq.X(x)])] * 4)
    c.insert_into_range([cirq.Z(x), cirq.CZ(x, y)], 2, 2)
    cirq.testing.assert_has_diagram(c, '\nx: ───X───X───Z───@───X───X───\n                  │\ny: ───────────────@───────────\n')
    c.insert_into_range([cirq.Y(y), cirq.Y(y), cirq.Y(y), cirq.CX(y, x)], 1, 4)
    cirq.testing.assert_has_diagram(c, '\nx: ───X───X───Z───@───X───X───X───\n                  │       │\ny: ───────Y───Y───@───Y───@───────\n')
    c.insert_into_range([cirq.H(y), cirq.H(y)], 6, 7)
    cirq.testing.assert_has_diagram(c, '\nx: ───X───X───Z───@───X───X───X───────\n                  │       │\ny: ───────Y───Y───@───Y───@───H───H───\n')
    c.insert_into_range([cirq.T(y)], 0, 1)
    cirq.testing.assert_has_diagram(c, '\nx: ───X───X───Z───@───X───X───X───────\n                  │       │\ny: ───T───Y───Y───@───Y───@───H───H───\n')
    with pytest.raises(IndexError):
        c.insert_into_range([cirq.CZ(x, y)], 10, 10)
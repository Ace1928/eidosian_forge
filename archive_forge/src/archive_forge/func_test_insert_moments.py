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
def test_insert_moments():
    q = cirq.NamedQubit('q')
    c = cirq.Circuit()
    m0 = cirq.Moment([cirq.X(q)])
    c.append(m0)
    assert list(c) == [m0]
    assert c[0] == m0
    m1 = cirq.Moment([cirq.Y(q)])
    c.append(m1)
    assert list(c) == [m0, m1]
    assert c[1] == m1
    m2 = cirq.Moment([cirq.Z(q)])
    c.insert(0, m2)
    assert list(c) == [m2, m0, m1]
    assert c[0] == m2
    assert c._moments == [m2, m0, m1]
    assert c._moments[0] == m2
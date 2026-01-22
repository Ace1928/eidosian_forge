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
def test_batch_replace():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    original = cirq.Circuit([cirq.Moment([cirq.X(a)]), cirq.Moment([cirq.Z(b)]), cirq.Moment([cirq.CZ(a, b)]), cirq.Moment([cirq.X(a), cirq.X(b)])])
    after = original.copy()
    after.batch_replace([])
    assert after == original
    after = original.copy()
    after.batch_replace([(0, cirq.X(a), cirq.Y(a))])
    assert after == cirq.Circuit([cirq.Moment([cirq.Y(a)]), cirq.Moment([cirq.Z(b)]), cirq.Moment([cirq.CZ(a, b)]), cirq.Moment([cirq.X(a), cirq.X(b)])])
    after = original.copy()
    with pytest.raises(IndexError):
        after.batch_replace([(500, cirq.X(a), cirq.Y(a))])
    assert after == original
    after = original.copy()
    with pytest.raises(ValueError):
        after.batch_replace([(0, cirq.Z(a), cirq.Y(a))])
    assert after == original
    after = original.copy()
    after.batch_replace([(0, cirq.X(a), cirq.Y(a)), (2, cirq.CZ(a, b), cirq.CNOT(a, b))])
    assert after == cirq.Circuit([cirq.Moment([cirq.Y(a)]), cirq.Moment([cirq.Z(b)]), cirq.Moment([cirq.CNOT(a, b)]), cirq.Moment([cirq.X(a), cirq.X(b)])])
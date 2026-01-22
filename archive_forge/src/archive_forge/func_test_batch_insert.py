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
def test_batch_insert():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    original = cirq.Circuit([cirq.Moment([cirq.X(a)]), cirq.Moment([]), cirq.Moment([cirq.CZ(a, b)]), cirq.Moment([cirq.X(a), cirq.X(b)])])
    after = original.copy()
    after.batch_insert([])
    assert after == original
    after = original.copy()
    after.batch_insert([(0, cirq.CZ(a, b)), (0, cirq.CNOT(a, b)), (1, cirq.Z(b))])
    assert after == cirq.Circuit([cirq.Moment([cirq.CNOT(a, b)]), cirq.Moment([cirq.CZ(a, b)]), cirq.Moment([cirq.X(a), cirq.Z(b)]), cirq.Moment(), cirq.Moment([cirq.CZ(a, b)]), cirq.Moment([cirq.X(a), cirq.X(b)])])
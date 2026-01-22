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
def test_insert_inline_near_start():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.Circuit([cirq.Moment(), cirq.Moment()])
    c.insert(1, cirq.X(a), strategy=cirq.InsertStrategy.INLINE)
    assert c == cirq.Circuit([cirq.Moment([cirq.X(a)]), cirq.Moment()])
    c.insert(1, cirq.Y(a), strategy=cirq.InsertStrategy.INLINE)
    assert c == cirq.Circuit([cirq.Moment([cirq.X(a)]), cirq.Moment([cirq.Y(a)]), cirq.Moment()])
    c.insert(0, cirq.Z(b), strategy=cirq.InsertStrategy.INLINE)
    assert c == cirq.Circuit([cirq.Moment([cirq.Z(b)]), cirq.Moment([cirq.X(a)]), cirq.Moment([cirq.Y(a)]), cirq.Moment()])
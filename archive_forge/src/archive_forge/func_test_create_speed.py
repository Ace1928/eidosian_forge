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
def test_create_speed():
    qs = 100
    moments = 500
    xs = [cirq.X(cirq.LineQubit(i)) for i in range(qs)]
    opa = [xs[i] for i in range(qs) for _ in range(moments)]
    t = time.perf_counter()
    c = cirq.Circuit(opa)
    assert len(c) == moments
    assert time.perf_counter() - t < 4
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
def test_freeze_not_relocate_moments():
    q = cirq.q(0)
    c = cirq.Circuit(cirq.X(q), cirq.measure(q))
    f = c.freeze()
    assert [mc is fc for mc, fc in zip(c, f)] == [True, True]
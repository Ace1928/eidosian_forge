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
def test_none_precision_diagram(circuit_cls):
    qa = cirq.NamedQubit('a')
    c = circuit_cls([cirq.Moment([cirq.X(qa) ** 0.4921875])])
    cirq.testing.assert_has_diagram(c, '\na: ---X^0.4921875---\n', use_unicode_characters=False, precision=None)
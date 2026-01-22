from typing import Tuple, List, cast
import re
import pytest
import sympy
import cirq
from cirq._compat import proper_repr
import numpy as np
def test_gateset_init():
    assert gateset.name == 'custom gateset'
    assert gateset.gates == frozenset([cirq.GateFamily(CustomX ** 0.5), cirq.GateFamily(cirq.testing.TwoQubitGate), CustomXGateFamily()])
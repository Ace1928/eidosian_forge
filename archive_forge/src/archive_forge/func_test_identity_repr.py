import itertools
from typing import Any
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
def test_identity_repr():
    assert repr(cirq.I) == 'cirq.I'
    assert repr(cirq.IdentityGate(5)) == 'cirq.IdentityGate(5)'
    assert repr(cirq.IdentityGate(qid_shape=(2, 3))) == 'cirq.IdentityGate(qid_shape=(2, 3))'
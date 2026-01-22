import itertools
from typing import Any
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
def test_identity_str():
    assert str(cirq.IdentityGate(1)) == 'I'
    assert str(cirq.IdentityGate(2)) == 'I(2)'
    assert str(cirq.IdentityGate(1, (3,))) == 'I'
    assert str(cirq.IdentityGate(2, (1, 2))) == 'I(2)'
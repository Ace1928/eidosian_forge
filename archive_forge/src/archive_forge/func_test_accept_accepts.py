import math
from unittest import mock
import pytest
from cirq_google.line.placement import optimization
def test_accept_accepts():
    assert optimization._accept(0.0, 0.0, 1.0)[0]
    assert optimization._accept(0.0, -0.1, 1.0)[0]
    assert optimization._accept(0.0, 1.0, 1.0)[0]
    assert optimization._accept(1.0 / math.e - 1e-09, 1.0, 1.0)[0]
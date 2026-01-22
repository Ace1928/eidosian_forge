from typing import Optional, Sequence, Union, Collection, Tuple, List
import pytest
import numpy as np
import cirq
from cirq.ops import control_values as cv
def test_assert_controlled_and_controlled_by_identical():
    cirq.testing.assert_controlled_and_controlled_by_identical(GoodGate())
    with pytest.raises(AssertionError):
        cirq.testing.assert_controlled_and_controlled_by_identical(BadGate())
    with pytest.raises(ValueError, match='len\\(num_controls\\) != len\\(control_values\\)'):
        cirq.testing.assert_controlled_and_controlled_by_identical(GoodGate(), num_controls=[1, 2], control_values=[(1,)])
    with pytest.raises(ValueError, match='len\\(control_values\\[1\\]\\) != num_controls\\[1\\]'):
        cirq.testing.assert_controlled_and_controlled_by_identical(GoodGate(), num_controls=[1, 2], control_values=[(1,), (1, 1, 1)])
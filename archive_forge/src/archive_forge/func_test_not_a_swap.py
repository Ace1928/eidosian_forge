import dataclasses
import pytest
import numpy as np
import sympy
import cirq
from cirq.transformers.eject_z import _is_swaplike
@pytest.mark.parametrize('exponent', (0, 2, 1.1, -2, -1.6))
def test_not_a_swap(exponent):
    a, b = cirq.LineQubit.range(2)
    assert not _is_swaplike(cirq.SWAP(a, b) ** exponent)
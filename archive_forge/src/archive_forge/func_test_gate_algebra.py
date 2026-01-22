from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
@pytest.mark.parametrize('expression, expected_result', ((cirq.X * 2, 2 * cirq.X), (cirq.Y * 2, cirq.Y + cirq.Y), (cirq.Z - cirq.Z + cirq.Z, cirq.Z.wrap_in_linear_combination()), (1j * cirq.S * 1j, -cirq.S), (cirq.CZ * 1, cirq.CZ / 1), (-cirq.CSWAP * 1j, cirq.CSWAP / 1j), (cirq.TOFFOLI * 0.5, cirq.TOFFOLI / 2)))
def test_gate_algebra(expression, expected_result):
    assert expression == expected_result
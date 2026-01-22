import itertools
from typing import Optional
from unittest import mock
import pytest
import cirq
def test_decompose_general():
    a, b, c = cirq.LineQubit.range(3)
    no_method = NoMethod()
    assert cirq.decompose(no_method) == [no_method]
    assert cirq.decompose([cirq.SWAP(a, b), cirq.SWAP(a, b)]) == 2 * cirq.decompose(cirq.SWAP(a, b))
    ops = (cirq.TOFFOLI(a, b, c), cirq.H(a), cirq.CZ(a, c))
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(cirq.Circuit(ops), cirq.Circuit(cirq.decompose(ops)), atol=1e-08)
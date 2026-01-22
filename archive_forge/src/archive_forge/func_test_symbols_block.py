import dataclasses
import pytest
import numpy as np
import sympy
import cirq
from cirq.transformers.eject_z import _is_swaplike
@pytest.mark.parametrize('sym', [sympy.Symbol('a'), sympy.Symbol('a') + 1])
def test_symbols_block(sym):
    q = cirq.NamedQubit('q')
    assert_optimizes(before=cirq.Circuit([cirq.Moment([cirq.Z(q)]), cirq.Moment([cirq.Z(q) ** sym]), cirq.Moment([cirq.Z(q) ** 0.25])]), expected=cirq.Circuit([cirq.Moment(), cirq.Moment([cirq.Z(q) ** sym]), cirq.Moment([cirq.Z(q) ** 1.25])]))
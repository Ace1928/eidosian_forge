import dataclasses
import pytest
import numpy as np
import sympy
import cirq
from cirq.transformers.eject_z import _is_swaplike
def test_z_pushes_past_cz():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    assert_optimizes(before=cirq.Circuit([cirq.Moment([cirq.Z(a) ** 0.5]), cirq.Moment([cirq.CZ(a, b) ** 0.25])]), expected=cirq.Circuit([cirq.Moment(), cirq.Moment([cirq.CZ(a, b) ** 0.25]), cirq.Moment([cirq.Z(a) ** 0.5])]))
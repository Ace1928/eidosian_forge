import dataclasses
import pytest
import numpy as np
import sympy
import cirq
from cirq.transformers.eject_z import _is_swaplike
def test_single_z_stays():
    q = cirq.NamedQubit('q')
    assert_optimizes(before=cirq.Circuit([cirq.Moment([cirq.Z(q) ** 0.5])]), expected=cirq.Circuit([cirq.Moment([cirq.Z(q) ** 0.5])]))
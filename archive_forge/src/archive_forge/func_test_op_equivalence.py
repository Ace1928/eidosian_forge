import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_op_equivalence():
    a, b = cirq.LineQubit.range(2)
    various_x = [cirq.X(a), cirq.PauliString({a: cirq.X}), cirq.PauliString([cirq.X.on(a)]), cirq.SingleQubitPauliStringGateOperation(cirq.X, a), cirq.GateOperation(cirq.X, [a])]
    for x in various_x:
        cirq.testing.assert_equivalent_repr(x)
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(*various_x)
    eq.add_equality_group(cirq.Y(a), cirq.PauliString({a: cirq.Y}))
    eq.add_equality_group(-cirq.PauliString({a: cirq.X}))
    eq.add_equality_group(cirq.Z(a), cirq.PauliString({a: cirq.Z}))
    eq.add_equality_group(cirq.Z(b), cirq.PauliString({b: cirq.Z}))
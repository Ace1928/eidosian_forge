import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_op_product():
    a, b = cirq.LineQubit.range(2)
    assert cirq.X(a) * cirq.X(b) == cirq.PauliString({a: cirq.X, b: cirq.X})
    assert cirq.X(a) * cirq.Y(b) == cirq.PauliString({a: cirq.X, b: cirq.Y})
    assert cirq.Z(a) * cirq.Y(b) == cirq.PauliString({a: cirq.Z, b: cirq.Y})
    assert cirq.X(a) * cirq.X(a) == cirq.PauliString()
    assert cirq.X(a) * cirq.Y(a) == 1j * cirq.PauliString({a: cirq.Z})
    assert cirq.Y(a) * cirq.Z(b) * cirq.X(a) == -1j * cirq.PauliString({a: cirq.Z, b: cirq.Z})
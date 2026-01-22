from typing import cast, Iterable
import dataclasses
import numpy as np
import pytest
import sympy
import cirq
def test_zero_x_rotation():
    a = cirq.NamedQubit('a')
    assert_optimizes(before=quick_circuit([cirq.rx(0)(a)]), expected=quick_circuit([cirq.rx(0)(a)]))
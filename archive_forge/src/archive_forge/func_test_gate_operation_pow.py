import collections.abc
import pathlib
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_gate_operation_pow():
    Y = cirq.Y
    q = cirq.NamedQubit('q')
    assert (Y ** 0.5)(q) == Y(q) ** 0.5
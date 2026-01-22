import collections.abc
import pathlib
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_cannot_remap_non_measurement_gate():
    a = cirq.LineQubit(0)
    op = cirq.X(a)
    assert cirq.with_measurement_key_mapping(op, {'m': 'k'}) is NotImplemented
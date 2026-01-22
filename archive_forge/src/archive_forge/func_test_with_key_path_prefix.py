import collections.abc
import pathlib
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_with_key_path_prefix():
    a = cirq.LineQubit(0)
    op = cirq.measure(a, key='m')
    remap_op = cirq.with_key_path_prefix(op, ('a', 'b'))
    assert cirq.measurement_key_names(remap_op) == {'a:b:m'}
    assert cirq.with_key_path_prefix(remap_op, tuple()) is remap_op
    assert cirq.with_key_path_prefix(op, tuple()) is op
    assert cirq.with_key_path_prefix(cirq.X(a), ('a', 'b')) is NotImplemented
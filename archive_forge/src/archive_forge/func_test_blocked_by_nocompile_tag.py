from typing import cast, Iterable
import dataclasses
import numpy as np
import pytest
import sympy
import cirq
def test_blocked_by_nocompile_tag():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    assert_optimizes(before=quick_circuit([cirq.X(a)], [cirq.CZ(a, b).with_tags('nocompile')], [cirq.X(a)]), expected=quick_circuit([cirq.X(a)], [cirq.CZ(a, b).with_tags('nocompile')], [cirq.X(a)]), with_context=True)
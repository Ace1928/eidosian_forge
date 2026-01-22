import itertools
import pytest
import sympy
import cirq
def test_to_sweeps_iterable_sweeps():
    sweeps = [cirq.Linspace('a', 0, 1, 10), cirq.Linspace('b', 0, 1, 10)]
    assert cirq.study.to_sweeps(sweeps) == sweeps
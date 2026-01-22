import itertools
import pytest
import sympy
import cirq
def test_to_sweeps_sweep():
    sweep = cirq.Linspace('a', 0, 1, 10)
    assert cirq.study.to_sweeps(sweep) == [sweep]
import itertools
import pytest
import sympy
import cirq
def test_to_sweeps_single():
    resolver = cirq.ParamResolver({})
    assert cirq.study.to_sweeps(resolver) == [cirq.UnitSweep]
    assert cirq.study.to_sweeps({}) == [cirq.UnitSweep]
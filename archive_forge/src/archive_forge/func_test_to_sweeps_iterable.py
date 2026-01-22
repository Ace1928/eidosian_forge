import itertools
import pytest
import sympy
import cirq
def test_to_sweeps_iterable():
    resolvers = [cirq.ParamResolver({'a': 2}), cirq.ParamResolver({'a': 1})]
    sweeps = [cirq.study.Zip(cirq.Points('a', [2])), cirq.study.Zip(cirq.Points('a', [1]))]
    assert cirq.study.to_sweeps(resolvers) == sweeps
    assert cirq.study.to_sweeps([{'a': 2}, {'a': 1}]) == sweeps
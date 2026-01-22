from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_tagged_operation_resolves_parameterized_tags(resolve_fn):
    q = cirq.GridQubit(0, 0)
    tag = ParameterizableTag(sympy.Symbol('t'))
    assert cirq.is_parameterized(tag)
    assert cirq.parameter_names(tag) == {'t'}
    op = cirq.Z(q).with_tags(tag)
    assert cirq.is_parameterized(op)
    assert cirq.parameter_names(op) == {'t'}
    resolved_op = resolve_fn(op, {'t': 10})
    assert resolved_op == cirq.Z(q).with_tags(ParameterizableTag(10))
    assert not cirq.is_parameterized(resolved_op)
    assert cirq.parameter_names(resolved_op) == set()
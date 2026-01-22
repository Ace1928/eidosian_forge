import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_resolve_parameters_no_change():
    a, b = cirq.LineQubit.range(2)
    moment = cirq.Moment(cirq.X(a), cirq.Y(b))
    resolved_moment = cirq.resolve_parameters(moment, cirq.ParamResolver({'v': 0.1, 'w': 0.2}))
    assert resolved_moment is moment
    moment = cirq.Moment(cirq.X(a) ** sympy.Symbol('v'), cirq.Y(b) ** sympy.Symbol('w'))
    resolved_moment = cirq.resolve_parameters(moment, cirq.ParamResolver({}))
    assert resolved_moment is moment
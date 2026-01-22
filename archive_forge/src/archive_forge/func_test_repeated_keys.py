import numpy as np
import pytest
import sympy
import cirq
def test_repeated_keys():
    q0, q1, q2 = cirq.LineQubit.range(3)
    c = cirq.Circuit(cirq.measure(q0, key='a'), cirq.measure(q1, q2, key='b'), cirq.measure(q0, key='a'), cirq.measure(q1, q2, key='b'), cirq.measure(q1, q2, key='b'))
    result = cirq.ZerosSampler().run(c, repetitions=10)
    assert result.records['a'].shape == (10, 2, 1)
    assert result.records['b'].shape == (10, 3, 2)
    c2 = cirq.Circuit(cirq.measure(q0, key='a'), cirq.measure(q1, q2, key='a'))
    with pytest.raises(ValueError, match='Different qid shapes for repeated measurement'):
        cirq.ZerosSampler().run(c2, repetitions=10)
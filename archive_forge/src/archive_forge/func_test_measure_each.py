import numpy as np
import pytest
import cirq
def test_measure_each():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    assert cirq.measure_each() == []
    assert cirq.measure_each([]) == []
    assert cirq.measure_each(a) == [cirq.measure(a)]
    assert cirq.measure_each([a]) == [cirq.measure(a)]
    assert cirq.measure_each(a, b) == [cirq.measure(a), cirq.measure(b)]
    assert cirq.measure_each([a, b]) == [cirq.measure(a), cirq.measure(b)]
    qubit_generator = (q for q in (a, b))
    assert cirq.measure_each(qubit_generator) == [cirq.measure(a), cirq.measure(b)]
    assert cirq.measure_each(a.with_dimension(3), b.with_dimension(3)) == [cirq.measure(a.with_dimension(3)), cirq.measure(b.with_dimension(3))]
    assert cirq.measure_each(a, b, key_func=lambda e: e.name + '!') == [cirq.measure(a, key='a!'), cirq.measure(b, key='b!')]
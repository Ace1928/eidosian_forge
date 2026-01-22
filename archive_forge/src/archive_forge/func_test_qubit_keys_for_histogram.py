import collections
import numpy as np
import pandas as pd
import pytest
import cirq
import cirq.testing
from cirq.study.result import _pack_digits
def test_qubit_keys_for_histogram():
    a, b, c = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.measure(a, b), cirq.X(c), cirq.measure(c))
    results = cirq.Simulator().run(program=circuit, repetitions=100)
    with pytest.raises(KeyError):
        _ = results.histogram(key=a)
    assert results.histogram(key=[a, b]) == collections.Counter({0: 100})
    assert results.histogram(key=c) == collections.Counter({True: 100})
    assert results.histogram(key=[c]) == collections.Counter({1: 100})
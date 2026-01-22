import numpy as np
import pytest
import cirq
import sympy
def test_incompatible_measurements():
    qs = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.measure(qs, key='key'), cirq.measure(qs[0], key='key'))
    sim = cirq.ClassicalStateSimulator()
    with pytest.raises(ValueError):
        _ = sim.run(c)
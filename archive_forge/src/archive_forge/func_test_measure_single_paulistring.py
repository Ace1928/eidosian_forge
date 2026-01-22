import numpy as np
import pytest
import cirq
def test_measure_single_paulistring():
    q = cirq.LineQubit.range(3)
    ps = cirq.X(q[0]) * cirq.Y(q[1]) * cirq.Z(q[2])
    assert cirq.measure_single_paulistring(ps, key='a') == cirq.PauliMeasurementGate(ps.values(), key='a').on(*ps.keys())
    ps_neg = -cirq.Y(cirq.LineQubit(0)) * cirq.Y(cirq.LineQubit(1))
    assert cirq.measure_single_paulistring(ps_neg, key='1').gate == cirq.PauliMeasurementGate(cirq.DensePauliString('YY', coefficient=-1), key='1')
    with pytest.raises(ValueError, match='should be an instance of cirq.PauliString'):
        _ = cirq.measure_single_paulistring(cirq.I(q[0]) * cirq.I(q[1]))
    with pytest.raises(ValueError, match='should be an instance of cirq.PauliString'):
        _ = cirq.measure_single_paulistring(q)
    with pytest.raises(ValueError, match='must have a coefficient'):
        _ = cirq.measure_single_paulistring(-2 * ps)
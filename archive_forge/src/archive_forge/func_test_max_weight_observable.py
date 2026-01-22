import pytest
import sympy
import cirq
from cirq.work.observable_settings import _max_weight_state, _max_weight_observable, _hashable_param
from cirq.work import InitObsSetting, observables_to_settings, _MeasurementSpec
def test_max_weight_observable():
    q0, q1 = cirq.LineQubit.range(2)
    observables = [cirq.X(q0), cirq.X(q1)]
    assert _max_weight_observable(observables) == cirq.X(q0) * cirq.X(q1)
    observables = [cirq.X(q0), cirq.X(q1), cirq.Z(q1)]
    assert _max_weight_observable(observables) is None
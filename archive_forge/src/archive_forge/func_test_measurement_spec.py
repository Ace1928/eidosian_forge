import pytest
import sympy
import cirq
from cirq.work.observable_settings import _max_weight_state, _max_weight_observable, _hashable_param
from cirq.work import InitObsSetting, observables_to_settings, _MeasurementSpec
def test_measurement_spec():
    q0, q1 = cirq.LineQubit.range(2)
    setting = InitObsSetting(init_state=cirq.KET_ZERO(q0) * cirq.KET_ZERO(q1), observable=cirq.X(q0) * cirq.Y(q1))
    meas_spec = _MeasurementSpec(max_setting=setting, circuit_params={'beta': 0.123, 'gamma': 0.456})
    meas_spec2 = _MeasurementSpec(max_setting=setting, circuit_params={'beta': 0.123, 'gamma': 0.456})
    assert hash(meas_spec) == hash(meas_spec2)
    cirq.testing.assert_equivalent_repr(meas_spec)
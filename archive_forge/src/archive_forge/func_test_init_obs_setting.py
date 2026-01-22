import pytest
import sympy
import cirq
from cirq.work.observable_settings import _max_weight_state, _max_weight_observable, _hashable_param
from cirq.work import InitObsSetting, observables_to_settings, _MeasurementSpec
def test_init_obs_setting():
    q0, q1 = cirq.LineQubit.range(2)
    setting = InitObsSetting(init_state=cirq.KET_ZERO(q0) * cirq.KET_ZERO(q1), observable=cirq.X(q0) * cirq.Y(q1))
    assert str(setting) == '+Z(q(0)) * +Z(q(1)) → X(q(0))*Y(q(1))'
    assert eval(repr(setting)) == setting
    with pytest.raises(ValueError):
        setting = InitObsSetting(init_state=cirq.KET_ZERO(q0), observable=cirq.X(q0) * cirq.Y(q1))
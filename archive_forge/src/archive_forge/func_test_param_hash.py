import pytest
import sympy
import cirq
from cirq.work.observable_settings import _max_weight_state, _max_weight_observable, _hashable_param
from cirq.work import InitObsSetting, observables_to_settings, _MeasurementSpec
def test_param_hash():
    params1 = [('beta', 1.23), ('gamma', 4.56)]
    params2 = [('beta', 1.23), ('gamma', 4.56)]
    params3 = [('beta', 1.24), ('gamma', 4.57)]
    params4 = [('beta', 1.23 + 0.01j), ('gamma', 4.56 + 0.01j)]
    params5 = [('beta', 1.23 + 0.01j), ('gamma', 4.56 + 0.01j)]
    assert _hashable_param(params1) == _hashable_param(params1)
    assert hash(_hashable_param(params1)) == hash(_hashable_param(params1))
    assert _hashable_param(params1) == _hashable_param(params2)
    assert hash(_hashable_param(params1)) == hash(_hashable_param(params2))
    assert _hashable_param(params1) != _hashable_param(params3)
    assert hash(_hashable_param(params1)) != hash(_hashable_param(params3))
    assert _hashable_param(params1) != _hashable_param(params4)
    assert hash(_hashable_param(params1)) != hash(_hashable_param(params4))
    assert _hashable_param(params4) == _hashable_param(params5)
    assert hash(_hashable_param(params4)) == hash(_hashable_param(params5))
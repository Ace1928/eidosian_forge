import pytest
import cirq
@pytest.mark.parametrize('gate', [ReturnsStr(), ReturnsObj()])
def test_measurement_key_obj(gate):
    assert isinstance(cirq.measurement_key_obj(gate), cirq.MeasurementKey)
    assert cirq.measurement_key_obj(gate) == cirq.MeasurementKey(name='door locker')
    assert cirq.measurement_key_obj(gate) == 'door locker'
    assert cirq.measurement_key_obj(gate, None) == 'door locker'
    assert cirq.measurement_key_obj(gate, NotImplemented) == 'door locker'
    assert cirq.measurement_key_obj(gate, 'a') == 'door locker'
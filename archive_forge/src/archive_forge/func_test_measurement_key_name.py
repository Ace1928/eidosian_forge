import pytest
import cirq
@pytest.mark.parametrize('gate', [ReturnsStr(), ReturnsObj()])
def test_measurement_key_name(gate):
    assert isinstance(cirq.measurement_key_name(gate), str)
    assert cirq.measurement_key_name(gate) == 'door locker'
    assert cirq.measurement_key_obj(gate) == cirq.MeasurementKey(name='door locker')
    assert cirq.measurement_key_name(gate, None) == 'door locker'
    assert cirq.measurement_key_name(gate, NotImplemented) == 'door locker'
    assert cirq.measurement_key_name(gate, 'a') == 'door locker'
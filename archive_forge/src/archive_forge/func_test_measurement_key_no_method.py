import pytest
import cirq
@pytest.mark.parametrize('key_method', [cirq.measurement_key_name, cirq.measurement_key_obj])
def test_measurement_key_no_method(key_method):

    class NoMethod:
        pass
    with pytest.raises(TypeError, match='no measurement keys'):
        key_method(NoMethod())
    with pytest.raises(ValueError, match='multiple measurement keys'):
        key_method(cirq.Circuit(cirq.measure(cirq.LineQubit(0), key='a'), cirq.measure(cirq.LineQubit(0), key='b')))
    assert key_method(NoMethod(), None) is None
    assert key_method(NoMethod(), NotImplemented) is NotImplemented
    assert key_method(NoMethod(), 'a') == 'a'
    assert key_method(cirq.X, None) is None
    assert key_method(cirq.X(cirq.LineQubit(0)), None) is None
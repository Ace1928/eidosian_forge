import pytest
import cirq
def test_pow_error():
    with pytest.raises(TypeError, match='returned NotImplemented'):
        _ = cirq.pow(ReturnsNotImplemented(), 3)
    with pytest.raises(TypeError, match='no __pow__ method'):
        _ = cirq.pow(NoMethod(), 3)
import pytest
import numpy as np
import cirq
@pytest.mark.parametrize('val', (NoMethod(), ReturnsNotImplemented(), ReturnsNotImplementedUnitary()))
def test_objects_with_no_mixture(val):
    with pytest.raises(TypeError, match='mixture'):
        _ = cirq.mixture(val)
    assert cirq.mixture(val, None) is None
    assert cirq.mixture(val, NotImplemented) is NotImplemented
    default = ((0.4, 'a'), (0.6, 'b'))
    assert cirq.mixture(val, default) == default
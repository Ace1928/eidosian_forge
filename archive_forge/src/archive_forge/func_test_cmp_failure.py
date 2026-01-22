import pytest
import cirq
def test_cmp_failure():
    with pytest.raises(TypeError, match='not supported between instances'):
        _ = 0 < cirq.LineQubit(1)
    with pytest.raises(TypeError, match='not supported between instances'):
        _ = cirq.LineQubit(1) < 0
    with pytest.raises(TypeError, match='not supported between instances'):
        _ = 0 < cirq.LineQid(1, 3)
    with pytest.raises(TypeError, match='not supported between instances'):
        _ = cirq.LineQid(1, 3) < 0
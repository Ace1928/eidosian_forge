import pytest
import sympy
import cirq
def test_slice_access_error():
    sweep = cirq.Points('a', [1, 2, 3])
    with pytest.raises(TypeError, match="<class 'str'>"):
        _ = sweep['junk']
    with pytest.raises(IndexError):
        _ = sweep[4]
    with pytest.raises(IndexError):
        _ = sweep[-4]
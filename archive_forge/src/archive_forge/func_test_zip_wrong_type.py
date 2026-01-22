import pytest
import sympy
import cirq
def test_zip_wrong_type():
    with pytest.raises(TypeError):
        _ = cirq.Linspace('a', 0, 9, 10) + 2
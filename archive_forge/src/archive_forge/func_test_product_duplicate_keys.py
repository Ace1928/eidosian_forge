import pytest
import sympy
import cirq
def test_product_duplicate_keys():
    with pytest.raises(ValueError):
        _ = cirq.Linspace('a', 0, 9, 10) * cirq.Linspace('a', 0, 10, 11)
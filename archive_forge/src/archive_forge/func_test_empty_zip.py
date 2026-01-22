import pytest
import sympy
import cirq
def test_empty_zip():
    assert len(cirq.ZipLongest()) == 0
    with pytest.raises(ValueError, match='non-empty'):
        _ = cirq.ZipLongest(cirq.Points('e', []), cirq.Points('a', [1, 2, 3]))
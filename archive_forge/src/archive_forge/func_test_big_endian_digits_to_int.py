import pytest
import cirq
def test_big_endian_digits_to_int():
    with pytest.raises(ValueError, match='len\\(base\\)'):
        _ = cirq.big_endian_digits_to_int([1, 2, 3], base=[2, 3, 5, 7])
    with pytest.raises(ValueError, match='Out of range'):
        _ = cirq.big_endian_digits_to_int([105, 106, 107], base=4)
    assert cirq.big_endian_digits_to_int([0, 1], base=102) == 1
    assert cirq.big_endian_digits_to_int([1, 0], base=102) == 102
    assert cirq.big_endian_digits_to_int([1, 0], base=[5, 7]) == 7
    assert cirq.big_endian_digits_to_int([0, 1], base=[5, 7]) == 1
    assert cirq.big_endian_digits_to_int([1, 2, 3, 4], base=[2, 3, 5, 7]) == 200
    assert cirq.big_endian_digits_to_int([1, 2, 3, 4], base=10) == 1234
    assert cirq.big_endian_digits_to_int((e for e in [1, 2, 3, 4]), base=(e for e in [2, 3, 5, 7])) == 200
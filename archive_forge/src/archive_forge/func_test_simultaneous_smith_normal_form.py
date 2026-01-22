from ..pari import pari
import fractions
def test_simultaneous_smith_normal_form(in1, in2, u0, u1, u2, d1, d2):
    _assert_at_most_one_zero_entry_per_row_or_column(d1)
    _assert_at_most_one_zero_entry_per_row_or_column(d2)
    assert has_full_rank(u0)
    assert has_full_rank(u1)
    assert has_full_rank(u2)
    assert _change_coordinates(u0, u1, in1) == d1
    assert _change_coordinates(u1, u2, in2) == d2
    assert is_matrix_zero(matrix_mult(in1, in2))
    assert is_matrix_zero(matrix_mult(d1, d2))
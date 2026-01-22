import pytest
import rpy2.robjects.packages_utils as p_u
def test_default_symbol_r2python():
    test_values = (('foo', 'foo'), ('foo.bar', 'foo_bar'), ('foo_bar', 'foo_bar'))
    for provided, expected in test_values:
        assert expected == p_u.default_symbol_r2python(provided)
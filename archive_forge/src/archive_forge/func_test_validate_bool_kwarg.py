import pytest
from pandas.util._validators import (
@pytest.mark.parametrize('name', ['inplace', 'copy'])
@pytest.mark.parametrize('value', [True, False, None])
def test_validate_bool_kwarg(name, value):
    assert validate_bool_kwarg(value, name) == value
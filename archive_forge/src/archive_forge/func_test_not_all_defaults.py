import pytest
from pandas.util._validators import validate_args
@pytest.mark.parametrize('i', range(1, 3))
def test_not_all_defaults(i, _fname):
    bad_arg = 'foo'
    msg = f"the '{bad_arg}' parameter is not supported in the pandas implementation of {_fname}\\(\\)"
    compat_args = {'foo': 2, 'bar': -1, 'baz': 3}
    arg_vals = (1, -1, 3)
    with pytest.raises(ValueError, match=msg):
        validate_args(_fname, arg_vals[:i], 2, compat_args)
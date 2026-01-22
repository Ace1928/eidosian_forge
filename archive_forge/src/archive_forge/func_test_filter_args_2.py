import functools
from joblib.func_inspect import filter_args, get_func_name, get_func_code
from joblib.func_inspect import _clean_win_chars, format_signature
from joblib.memory import Memory
from joblib.test.common import with_numpy
from joblib.testing import fixture, parametrize, raises
def test_filter_args_2():
    assert filter_args(j, [], (1, 2), {'ee': 2}) == {'x': 1, 'y': 2, '**': {'ee': 2}}
    ff = functools.partial(f, 1)
    assert filter_args(ff, [], (1,)) == {'*': [1], '**': {}}
    assert filter_args(ff, ['y'], (1,)) == {'*': [1], '**': {}}
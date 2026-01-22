import functools
from joblib.func_inspect import filter_args, get_func_name, get_func_code
from joblib.func_inspect import _clean_win_chars, format_signature
from joblib.memory import Memory
from joblib.test.common import with_numpy
from joblib.testing import fixture, parametrize, raises
def test_func_inspect_errors():
    assert get_func_name('a'.lower)[-1] == 'lower'
    assert get_func_code('a'.lower)[1:] == (None, -1)
    ff = lambda x: x
    assert get_func_name(ff, win_characters=False)[-1] == '<lambda>'
    assert get_func_code(ff)[1] == __file__.replace('.pyc', '.py')
    ff.__module__ = '__main__'
    assert get_func_name(ff, win_characters=False)[-1] == '<lambda>'
    assert get_func_code(ff)[1] == __file__.replace('.pyc', '.py')
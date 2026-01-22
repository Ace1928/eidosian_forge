import functools
from joblib.func_inspect import filter_args, get_func_name, get_func_code
from joblib.func_inspect import _clean_win_chars, format_signature
from joblib.memory import Memory
from joblib.test.common import with_numpy
from joblib.testing import fixture, parametrize, raises
def test_func_name_collision_on_inner_func():

    def f():

        def inner_func():
            return
        return get_func_name(inner_func)

    def g():

        def inner_func():
            return
        return get_func_name(inner_func)
    module, name = f()
    other_module, other_name = g()
    assert name == other_name
    assert module != other_module
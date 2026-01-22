import functools
from joblib.func_inspect import filter_args, get_func_name, get_func_code
from joblib.func_inspect import _clean_win_chars, format_signature
from joblib.memory import Memory
from joblib.test.common import with_numpy
from joblib.testing import fixture, parametrize, raises
@fixture(scope='module')
def cached_func(tmpdir_factory):
    cachedir = tmpdir_factory.mktemp('joblib_test_func_inspect')
    mem = Memory(cachedir.strpath)

    @mem.cache
    def cached_func_inner(x):
        return x
    return cached_func_inner
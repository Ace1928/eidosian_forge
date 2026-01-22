import functools
from joblib.func_inspect import filter_args, get_func_name, get_func_code
from joblib.func_inspect import _clean_win_chars, format_signature
from joblib.memory import Memory
from joblib.test.common import with_numpy
from joblib.testing import fixture, parametrize, raises
def test_format_signature_long_arguments():
    shortening_threshold = 1500
    shortening_target = 700 + 10
    arg = 'a' * shortening_threshold
    _, signature = format_signature(h, arg)
    assert len(signature) < shortening_target
    nb_args = 5
    args = [arg for _ in range(nb_args)]
    _, signature = format_signature(h, *args)
    assert len(signature) < shortening_target * nb_args
    kwargs = {str(i): arg for i, arg in enumerate(args)}
    _, signature = format_signature(h, **kwargs)
    assert len(signature) < shortening_target * nb_args
    _, signature = format_signature(h, *args, **kwargs)
    assert len(signature) < shortening_target * 2 * nb_args
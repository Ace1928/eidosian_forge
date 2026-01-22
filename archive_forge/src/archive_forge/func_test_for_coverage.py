import functools
import toolz._signatures as _sigs
from toolz._signatures import builtins, _is_valid_args, _is_partial_args
def test_for_coverage():
    assert _sigs._is_arity(1, 1) is None
    assert _sigs._is_arity(1, all)
    assert _sigs._has_varargs(None) is None
    assert _sigs._has_keywords(None) is None
    assert _sigs._num_required_args(None) is None
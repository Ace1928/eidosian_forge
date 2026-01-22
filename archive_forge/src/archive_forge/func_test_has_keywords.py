import functools
import inspect
import itertools
import operator
import toolz
from toolz.functoolz import (curry, is_valid_args, is_partial_args, is_arity,
from toolz._signatures import builtins
import toolz._signatures as _sigs
from toolz.utils import raises
def test_has_keywords():
    assert has_keywords(lambda: None) is False
    assert has_keywords(lambda x: None) is False
    assert has_keywords(lambda x=1: None)
    assert has_keywords(lambda **kwargs: None)
    assert has_keywords(int)
    assert has_keywords(sorted)
    assert has_keywords(max)
    assert has_keywords(map) is False
    assert has_keywords(bytearray) is None
import numpy
import pytest
from hypothesis import given
from thinc.api import Padded, Ragged, get_width
from thinc.types import ArgsKwargs
from thinc.util import (
from . import strategies
def test_convert_recursive():
    is_match = lambda obj: obj == 'foo'
    convert_item = lambda obj: obj.upper()
    obj = {'a': {('b', 'foo'): {'c': 'foo', 'd': ['foo', {'e': 'foo', 'f': (1, 'foo')}]}}}
    result = convert_recursive(is_match, convert_item, obj)
    assert result['a']['b', 'FOO']['c'] == 'FOO'
    assert result['a']['b', 'FOO']['d'] == ['FOO', {'e': 'FOO', 'f': (1, 'FOO')}]
    obj = {'a': ArgsKwargs(('foo', [{'b': 'foo'}]), {'a': ['x', 'foo']})}
    result = convert_recursive(is_match, convert_item, obj)
    assert result['a'].args == ('FOO', [{'b': 'FOO'}])
    assert result['a'].kwargs == {'a': ['x', 'FOO']}
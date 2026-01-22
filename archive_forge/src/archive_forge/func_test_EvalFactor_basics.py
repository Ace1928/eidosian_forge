import sys
import __future__
import inspect
import tokenize
import ast
import numbers
import six
from patsy import PatsyError
from patsy.util import PushbackAdapter, no_pickling, assert_no_pickling
from patsy.tokens import (pretty_untokenize, normalize_token_spacing,
from patsy.compat import call_and_wrap_exc
import patsy.builtins
def test_EvalFactor_basics():
    e = EvalFactor('a+b')
    assert e.code == 'a + b'
    assert e.name() == 'a + b'
    e2 = EvalFactor('a    +b', origin='asdf')
    assert e == e2
    assert hash(e) == hash(e2)
    assert e.origin is None
    assert e2.origin == 'asdf'
    assert_no_pickling(e)
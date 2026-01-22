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
def test_capture_obj_method_calls():
    assert capture_obj_method_calls('foo', 'a + foo.baz(bar) + b.c(d)') == [('foo.baz', 'foo.baz(bar)')]
    assert capture_obj_method_calls('b', 'a + foo.baz(bar) + b.c(d)') == [('b.c', 'b.c(d)')]
    assert capture_obj_method_calls('foo', 'foo.bar(foo.baz(quux))') == [('foo.bar', 'foo.bar(foo.baz(quux))'), ('foo.baz', 'foo.baz(quux)')]
    assert capture_obj_method_calls('bar', 'foo[bar.baz(x(z[asdf])) ** 2]') == [('bar.baz', 'bar.baz(x(z[asdf]))')]
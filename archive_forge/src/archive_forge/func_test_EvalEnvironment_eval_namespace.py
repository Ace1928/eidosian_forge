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
def test_EvalEnvironment_eval_namespace():
    env = EvalEnvironment([{'a': 1}])
    assert env.eval('2 * a') == 2
    assert env.eval('2 * a', inner_namespace={'a': 2}) == 4
    import pytest
    pytest.raises(NameError, env.eval, '2 * b')
    a = 3
    env2 = EvalEnvironment.capture(0)
    assert env2.eval('2 * a') == 6
    env3 = env.with_outer_namespace({'a': 10, 'b': 3})
    assert env3.eval('2 * a') == 2
    assert env3.eval('2 * b') == 6
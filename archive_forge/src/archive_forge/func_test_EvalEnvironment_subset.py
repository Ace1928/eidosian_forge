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
def test_EvalEnvironment_subset():
    env = EvalEnvironment([{'a': 1}, {'b': 2}, {'c': 3}])
    subset_a = env.subset(['a'])
    assert subset_a.eval('a') == 1
    import pytest
    pytest.raises(NameError, subset_a.eval, 'b')
    pytest.raises(NameError, subset_a.eval, 'c')
    subset_bc = env.subset(['b', 'c'])
    assert subset_bc.eval('b * c') == 6
    pytest.raises(NameError, subset_bc.eval, 'a')
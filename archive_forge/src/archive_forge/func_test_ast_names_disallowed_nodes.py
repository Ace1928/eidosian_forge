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
def test_ast_names_disallowed_nodes():
    import pytest

    def list_ast_names(code):
        return list(ast_names(code))
    pytest.raises(PatsyError, list_ast_names, 'lambda x: x + y')
    pytest.raises(PatsyError, list_ast_names, '[x + 1 for x in range(10)]')
    pytest.raises(PatsyError, list_ast_names, '(x + 1 for x in range(10))')
    if sys.version_info >= (2, 7):
        pytest.raises(PatsyError, list_ast_names, '{x: True for x in range(10)}')
        pytest.raises(PatsyError, list_ast_names, '{x + 1 for x in range(10)}')
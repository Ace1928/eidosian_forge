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
def ast_names(code):
    """Iterator that yields all the (ast) names in a Python expression.

    :arg code: A string containing a Python expression.
    """
    disallowed_ast_nodes = (ast.Lambda, ast.ListComp, ast.GeneratorExp)
    if sys.version_info >= (2, 7):
        disallowed_ast_nodes += (ast.DictComp, ast.SetComp)
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, disallowed_ast_nodes):
            raise PatsyError('Lambda, list/dict/set comprehension, generator expression in patsy formula not currently supported.')
        if isinstance(node, ast.Name):
            yield node.id
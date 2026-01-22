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
def replace_bare_funcalls(code, replacer):
    tokens = []
    for token_type, token, origin, props in annotated_tokens(code):
        if props['bare_ref'] and props['bare_funcall']:
            token = replacer(token)
        tokens.append((token_type, token))
    return pretty_untokenize(tokens)
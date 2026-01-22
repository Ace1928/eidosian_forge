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
def new_name_maker(token):
    value = eval_env.namespace.get(token)
    if hasattr(value, '__patsy_stateful_transform__'):
        obj_name = '_patsy_stobj%s__%s__' % (i[0], token)
        i[0] += 1
        obj = value.__patsy_stateful_transform__()
        state['transforms'][obj_name] = obj
        return obj_name + '.transform'
    else:
        return token
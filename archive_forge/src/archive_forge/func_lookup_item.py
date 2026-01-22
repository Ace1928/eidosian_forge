from textwrap import dedent
from types import CodeType
import six
from six.moves import builtins
from genshi.core import Markup
from genshi.template.astutil import ASTTransformer, ASTCodeGenerator, parse
from genshi.template.base import TemplateRuntimeError
from genshi.util import flatten
from genshi.compat import ast as _ast, _ast_Constant, get_code_params, \
@classmethod
def lookup_item(cls, obj, key):
    __traceback_hide__ = True
    if len(key) == 1:
        key = key[0]
    try:
        return obj[key]
    except (AttributeError, KeyError, IndexError, TypeError) as e:
        if isinstance(key, six.string_types):
            val = getattr(obj, key, UNDEFINED)
            if val is UNDEFINED:
                val = cls.undefined(key, owner=obj)
            return val
        raise
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
def lookup_name(cls, data, name):
    __traceback_hide__ = True
    val = data.get(name, UNDEFINED)
    if val is UNDEFINED:
        val = BUILTINS.get(name, val)
        if val is UNDEFINED:
            val = cls.undefined(name)
    return val
from gettext import NullTranslations
import os
import re
from functools import partial
from types import FunctionType
import six
from genshi.core import Attrs, Namespace, QName, START, END, TEXT, \
from genshi.template.base import DirectiveFactory, EXPR, SUB, _apply_directives
from genshi.template.directives import Directive, StripDirective
from genshi.template.markup import MarkupTemplate, EXEC
from genshi.compat import ast, IS_PYTHON2, _ast_Str, _ast_Str_value
@classmethod
def attach(cls, template, stream, value, namespaces, pos):
    if type(value) is dict:
        value = value.get('name')
    return super(ContextDirective, cls).attach(template, stream, value, namespaces, pos)
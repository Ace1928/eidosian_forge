from __future__ import absolute_import
import cython
from io import StringIO
import re
import sys
from unicodedata import lookup as lookup_unicodechar, category as unicode_category
from functools import partial, reduce
from .Scanning import PyrexScanner, FileSourceDescriptor, tentatively_scan
from . import Nodes
from . import ExprNodes
from . import Builtin
from . import StringEncoding
from .StringEncoding import EncodedString, bytes_literal, _unicode, _bytes
from .ModuleNode import ModuleNode
from .Errors import error, warning
from .. import Utils
from . import Future
from . import Options
def p_suite_with_docstring(s, ctx, with_doc_only=False):
    s.expect(':')
    doc = None
    if s.sy == 'NEWLINE':
        s.next()
        s.expect_indent()
        if with_doc_only:
            doc = p_doc_string(s)
        body = p_statement_list(s, ctx)
        s.expect_dedent()
    else:
        if ctx.api:
            s.error("'api' not allowed with this statement", fatal=False)
        if ctx.level in ('module', 'class', 'function', 'other'):
            body = p_simple_statement_list(s, ctx)
        else:
            body = p_pass_statement(s)
            s.expect_newline('Syntax error in declarations', ignore_semicolon=True)
    if not with_doc_only:
        doc, body = _extract_docstring(body)
    return (doc, body)
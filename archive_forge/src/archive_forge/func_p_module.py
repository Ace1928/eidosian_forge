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
def p_module(s, pxd, full_module_name, ctx=Ctx):
    pos = s.position()
    directive_comments = p_compiler_directive_comments(s)
    s.parse_comments = False
    if s.context.language_level is None:
        s.context.set_language_level('3str')
        if pos[0].filename:
            import warnings
            warnings.warn("Cython directive 'language_level' not set, using '3str' for now (Py3). This has changed from earlier releases! File: %s" % pos[0].filename, FutureWarning, stacklevel=1 if cython.compiled else 2)
    level = 'module_pxd' if pxd else 'module'
    doc = p_doc_string(s)
    body = p_statement_list(s, ctx(level=level), first_statement=1)
    if s.sy != 'EOF':
        s.error('Syntax error in statement [%s,%s]' % (repr(s.sy), repr(s.systring)))
    return ModuleNode(pos, doc=doc, body=body, full_module_name=full_module_name, directive_comments=directive_comments)
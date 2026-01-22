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
def p_c_enum_item(s, ctx, items):
    pos = s.position()
    name = p_ident(s)
    cname = p_opt_cname(s)
    if cname is None and ctx.namespace is not None:
        cname = ctx.namespace + '::' + name
    value = None
    if s.sy == '=':
        s.next()
        value = p_test(s)
    items.append(Nodes.CEnumDefItemNode(pos, name=name, cname=cname, value=value))
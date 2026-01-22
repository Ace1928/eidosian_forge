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
def p_from_import_statement(s, first_statement=0):
    pos = s.position()
    s.next()
    if s.sy in ('.', '...'):
        level = 0
        while s.sy in ('.', '...'):
            level += len(s.sy)
            s.next()
    else:
        level = None
    if level is not None and s.sy in ('import', 'cimport'):
        dotted_name_pos, dotted_name = (s.position(), s.context.intern_ustring(''))
    else:
        if level is None and Future.absolute_import in s.context.future_directives:
            level = 0
        dotted_name_pos, _, dotted_name, _ = p_dotted_name(s, as_allowed=False)
    if s.sy not in ('import', 'cimport'):
        s.error("Expected 'import' or 'cimport'")
    kind = s.sy
    s.next()
    is_cimport = kind == 'cimport'
    is_parenthesized = False
    if s.sy == '*':
        imported_names = [(s.position(), s.context.intern_ustring('*'), None)]
        s.next()
    else:
        if s.sy == '(':
            is_parenthesized = True
            s.next()
        imported_names = [p_imported_name(s)]
    while s.sy == ',':
        s.next()
        if is_parenthesized and s.sy == ')':
            break
        imported_names.append(p_imported_name(s))
    if is_parenthesized:
        s.expect(')')
    if dotted_name == '__future__':
        if not first_statement:
            s.error('from __future__ imports must occur at the beginning of the file')
        elif level:
            s.error('invalid syntax')
        else:
            for name_pos, name, as_name in imported_names:
                if name == 'braces':
                    s.error('not a chance', name_pos)
                    break
                try:
                    directive = getattr(Future, name)
                except AttributeError:
                    s.error('future feature %s is not defined' % name, name_pos)
                    break
                s.context.future_directives.add(directive)
        return Nodes.PassStatNode(pos)
    elif is_cimport:
        return Nodes.FromCImportStatNode(pos, module_name=dotted_name, relative_level=level, imported_names=imported_names)
    else:
        imported_name_strings = []
        items = []
        for name_pos, name, as_name in imported_names:
            imported_name_strings.append(ExprNodes.IdentifierStringNode(name_pos, value=name))
            items.append((name, ExprNodes.NameNode(name_pos, name=as_name or name)))
        import_list = ExprNodes.ListNode(imported_names[0][0], args=imported_name_strings)
        return Nodes.FromImportStatNode(pos, module=ExprNodes.ImportNode(dotted_name_pos, module_name=ExprNodes.IdentifierStringNode(pos, value=dotted_name), level=level, name_list=import_list), items=items)
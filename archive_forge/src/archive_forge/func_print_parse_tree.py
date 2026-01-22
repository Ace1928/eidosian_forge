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
def print_parse_tree(f, node, level, key=None):
    ind = '  ' * level
    if node:
        f.write(ind)
        if key:
            f.write('%s: ' % key)
        t = type(node)
        if t is tuple:
            f.write('(%s @ %s\n' % (node[0], node[1]))
            for i in range(2, len(node)):
                print_parse_tree(f, node[i], level + 1)
            f.write('%s)\n' % ind)
            return
        elif isinstance(node, Nodes.Node):
            try:
                tag = node.tag
            except AttributeError:
                tag = node.__class__.__name__
            f.write('%s @ %s\n' % (tag, node.pos))
            for name, value in node.__dict__.items():
                if name != 'tag' and name != 'pos':
                    print_parse_tree(f, value, level + 1, name)
            return
        elif t is list:
            f.write('[\n')
            for i in range(len(node)):
                print_parse_tree(f, node[i], level + 1)
            f.write('%s]\n' % ind)
            return
    f.write('%s%s\n' % (ind, node))
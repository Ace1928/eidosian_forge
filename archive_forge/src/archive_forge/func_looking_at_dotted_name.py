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
def looking_at_dotted_name(s):
    if s.sy == 'IDENT':
        name = s.systring
        name_pos = s.position()
        s.next()
        result = s.sy == '.'
        s.put_back(u'IDENT', name, name_pos)
        return result
    else:
        return 0
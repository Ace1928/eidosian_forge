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
def p_call(s, function):
    pos = s.position()
    positional_args, keyword_args = p_call_parse_args(s)
    if not keyword_args and len(positional_args) == 1 and isinstance(positional_args[0], list):
        return ExprNodes.SimpleCallNode(pos, function=function, args=positional_args[0])
    else:
        arg_tuple, keyword_dict = p_call_build_packed_args(pos, positional_args, keyword_args)
        return ExprNodes.GeneralCallNode(pos, function=function, positional_args=arg_tuple, keyword_args=keyword_dict)
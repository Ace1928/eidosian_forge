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
def p_call_build_packed_args(pos, positional_args, keyword_args):
    keyword_dict = None
    subtuples = [ExprNodes.TupleNode(pos, args=arg) if isinstance(arg, list) else ExprNodes.AsTupleNode(pos, arg=arg) for arg in positional_args]
    arg_tuple = reduce(partial(ExprNodes.binop_node, pos, '+'), subtuples)
    if keyword_args:
        kwargs = []
        dict_items = []
        for item in keyword_args:
            if isinstance(item, tuple):
                key, value = item
                dict_items.append(ExprNodes.DictItemNode(pos=key.pos, key=key, value=value))
            elif item.is_dict_literal:
                dict_items.extend(item.key_value_pairs)
            else:
                if dict_items:
                    kwargs.append(ExprNodes.DictNode(dict_items[0].pos, key_value_pairs=dict_items, reject_duplicates=True))
                    dict_items = []
                kwargs.append(item)
        if dict_items:
            kwargs.append(ExprNodes.DictNode(dict_items[0].pos, key_value_pairs=dict_items, reject_duplicates=True))
        if kwargs:
            if len(kwargs) == 1 and kwargs[0].is_dict_literal:
                keyword_dict = kwargs[0]
            else:
                keyword_dict = ExprNodes.MergedDictNode(pos, keyword_args=kwargs)
    return (arg_tuple, keyword_dict)
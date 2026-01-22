import re
import textwrap
from ast import literal_eval
from inspect import cleandoc
from weakref import WeakKeyDictionary
from parso.python import tree
from parso.cache import parser_cache
from parso import split_lines
def is_scope(node):
    t = node.type
    if t == 'comp_for':
        return node.children[1].type != 'sync_comp_for'
    return t in ('file_input', 'classdef', 'funcdef', 'lambdef', 'sync_comp_for')
import re
import textwrap
from ast import literal_eval
from inspect import cleandoc
from weakref import WeakKeyDictionary
from parso.python import tree
from parso.cache import parser_cache
from parso import split_lines
def get_parent_scope(node, include_flows=False):
    """
    Returns the underlying scope.
    """
    scope = node.parent
    if scope is None:
        return None
    while True:
        if is_scope(scope):
            if scope.type in ('classdef', 'funcdef', 'lambdef'):
                index = scope.children.index(':')
                if scope.children[index].start_pos >= node.start_pos:
                    if node.parent.type == 'param' and node.parent.name == node:
                        pass
                    elif node.parent.type == 'tfpdef' and node.parent.children[0] == node:
                        pass
                    else:
                        scope = scope.parent
                        continue
            return scope
        elif include_flows and isinstance(scope, tree.Flow):
            if not (scope.type == 'if_stmt' and any((n.start_pos <= node.start_pos < n.end_pos for n in scope.get_test_nodes()))):
                return scope
        scope = scope.parent
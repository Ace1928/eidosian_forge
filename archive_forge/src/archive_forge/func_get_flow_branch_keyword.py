import re
import textwrap
from ast import literal_eval
from inspect import cleandoc
from weakref import WeakKeyDictionary
from parso.python import tree
from parso.cache import parser_cache
from parso import split_lines
def get_flow_branch_keyword(flow_node, node):
    start_pos = node.start_pos
    if not flow_node.start_pos < start_pos <= flow_node.end_pos:
        raise ValueError('The node is not part of the flow.')
    keyword = None
    for i, child in enumerate(flow_node.children):
        if start_pos < child.start_pos:
            return keyword
        first_leaf = child.get_first_leaf()
        if first_leaf in _FLOW_KEYWORDS:
            keyword = first_leaf
    return None
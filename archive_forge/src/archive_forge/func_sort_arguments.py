from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import io
import logging
import re
import sys
from cmakelang import lex
from cmakelang import markup
from cmakelang.common import UserError
from cmakelang.lex import TokenType
from cmakelang.parse.argument_nodes import PositionalGroupNode
from cmakelang.parse.common import FlowType, NodeType, TreeNode
from cmakelang.parse.util import comment_is_tag
from cmakelang.parse import simple_nodes
def sort_arguments(children):
    argument_nodes = []
    for child in children:
        if child.pnode.node_type is NodeType.ARGUMENT:
            argument_nodes.append(child)
    argument_nodes = sorted(argument_nodes, key=filename_node_key)
    nodemap = {id(node): idx + 1 for idx, node in enumerate(argument_nodes)}
    sortlist = []
    keyidx = 0
    posidx = 0
    for child in children:
        if id(child) in nodemap:
            keyidx = nodemap[id(child)]
            posidx = 0
        sortlist.append((keyidx, posidx, child))
        posidx += 1
    return [child for _, _, child in sorted(sortlist)]
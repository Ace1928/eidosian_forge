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
def get_pathstr(node_path):
    """Given a list of nodes, construct a path string that can be used to
     identify that node in the tree."""
    pathcopy = []
    for node in node_path:
        if isinstance(node, BodyNode):
            continue
        if isinstance(node, ArgGroupNode):
            continue
        pathcopy.append(node)
    names = []
    for node in pathcopy:
        if isinstance(node, PargGroupNode):
            count = 0
            for sibling in node._parent.children:
                if sibling is node:
                    break
                if isinstance(sibling, PargGroupNode):
                    count += 1
            name = '{}[{}]'.format(node.name, count)
        else:
            name = node.name
        names.append(name)
    return '/'.join(names)
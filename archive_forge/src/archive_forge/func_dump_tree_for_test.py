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
def dump_tree_for_test(nodes, outfile=None, indent=None, increment=None):
    """
  Print a tree of node objects for debugging purposes
  """
    if indent is None:
        indent = ''
    if increment is None:
        increment = '  '
    if outfile is None:
        outfile = sys.stdout
    for node in nodes:
        outfile.write(indent)
        outfile.write('({}, {}, {}, {}, {}, ['.format(node.node_type, node.passno, node.position[0], node.position[1], node.colextent))
        if hasattr(node, 'children') and node.children:
            outfile.write('\n')
            dump_tree_for_test(node.children, outfile, indent + increment, increment)
            outfile.write(indent)
        outfile.write(']),\n')
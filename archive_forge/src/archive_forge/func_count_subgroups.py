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
def count_subgroups(children):
    """
  Count the number of positional or kwarg sub groups in an argument group.
  Ignore comments, and assert that no other types of children are found.
  """
    numgroups = 0
    for child in children:
        if child.node_type in (NodeType.KWARGGROUP, NodeType.PARGGROUP, NodeType.PARENGROUP):
            numgroups += 1
        elif child.node_type in (NodeType.COMMENT, NodeType.ONOFFSWITCH):
            continue
        else:
            raise ValueError('Unexpected node type {} as child of ArgGroupNode'.format(child.node_type))
    return numgroups
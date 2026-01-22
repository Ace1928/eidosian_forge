import enum
import logging
import re
from cmakelang.common import InternalError
from cmakelang.format.formatter import get_comment_lines
from cmakelang.lex import TokenType, Token
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.body_nodes import BodyNode, FlowControlNode
from cmakelang.parse.simple_nodes import CommentNode
from cmakelang.parse.common import NodeType, TreeNode
from cmakelang.parse.statement_node import StatementNode
from cmakelang.parse.util import get_min_npargs
from cmakelang.parse import variables
from cmakelang.parse.funs.set import SetFnNode
def find_nodes_in_subtree(subtree, nodetypes):
    """Return a generator that yields all nodes in the `subtree` which are of the
     given type(s) `nodetypes`.
  """
    if isinstance(subtree, (list, tuple)):
        queue = subtree
    else:
        queue = [subtree]
    while queue:
        node = queue.pop(0)
        if isinstance(node, nodetypes):
            yield node
        if isinstance(node, TreeNode):
            for child in node.children:
                queue.append(child)
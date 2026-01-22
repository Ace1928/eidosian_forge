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
def get_prefix_comment(prevchild):
    """
  Expect a sequence of COMMENT, WHITESPACE, <node>. If this sequence is true,
  return the comment node.
  """
    for idx in range(2):
        if not isinstance(prevchild[idx], TreeNode):
            return None
    if not prevchild[0].node_type is NodeType.WHITESPACE:
        return None
    newline_count = 0
    for token in prevchild[0].get_tokens():
        newline_count += token.spelling.count('\n')
    if newline_count > 1:
        return None
    if not prevchild[1].node_type is NodeType.COMMENT:
        return None
    return prevchild[1]
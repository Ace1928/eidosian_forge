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
def check_tree(self, node):
    if isinstance(node, Token):
        self.check_token(node)
        return
    if not isinstance(node, TreeNode):
        return
    self._node_stack.append(node)
    if node.node_type is NodeType.BODY:
        self.check_body(node)
    elif node.node_type is NodeType.FLOW_CONTROL:
        self.check_flow_control(node)
    elif isinstance(node, ArgGroupNode):
        self.check_arggroup(node)
    elif isinstance(node, StatementNode):
        self.check_statement(node)
    elif isinstance(node, PositionalGroupNode):
        self.check_positional_group(node)
    elif isinstance(node, CommentNode):
        self.check_comment(node)
    for child in node.children:
        self.check_tree(child)
    self._node_stack.pop(-1)
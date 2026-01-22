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
def check_flow_control(self, node):
    """Perform checks on a flowcontrol node."""
    stmt = node.children[0]
    funname = stmt.children[0].children[0].spelling.lower()
    if funname == 'function':
        self.check_fundef(stmt)
    elif funname == 'macro':
        self.check_macrodef(stmt)
    elif funname == 'foreach':
        self.check_foreach(stmt)
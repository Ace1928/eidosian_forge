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
def check_arggroup(self, node):
    _, local_ctx = self.context
    kwargs_seen = set()
    for child in node.children:
        if isinstance(child, TreeNode) and child.node_type is NodeType.KWARGGROUP:
            kwarg_token = child.get_semantic_tokens()[0]
            kwarg = kwarg_token.spelling.upper()
            if kwarg in ('AND', 'OR', 'COMMAND', 'PATTERN', 'REGEX'):
                continue
            if kwarg in kwargs_seen:
                local_ctx.record_lint('E1122', kwarg, location=kwarg_token.get_location())
            kwargs_seen.add(kwarg)
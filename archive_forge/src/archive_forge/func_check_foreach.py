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
def check_foreach(self, node):
    """Make sure that the loop variable matches the function argument
       pattern."""
    cfg, local_ctx = self.context
    tokens = node.get_semantic_tokens()
    tokens.pop(0)
    tokens.pop(0)
    token = tokens.pop(0)
    if not re.match(cfg.lint.argument_var_pattern, token.spelling):
        local_ctx.record_lint('C0103', 'argument', token.spelling, cfg.lint.argument_var_pattern, location=token.get_location())
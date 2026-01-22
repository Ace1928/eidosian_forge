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
def check_name_against_pattern(self, defn_node, pattern):
    """Check that a function or macro name matches the required pattern."""
    _, local_ctx = self.context
    tokens = defn_node.get_semantic_tokens()
    funname = tokens.pop(0).spelling
    tokens.pop(0)
    token = tokens.pop(0)
    if not re.match(pattern, token.spelling):
        local_ctx.record_lint('C0103', funname.lower(), token.spelling, pattern, location=token.get_location())
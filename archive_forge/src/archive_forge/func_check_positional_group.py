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
def check_positional_group(self, node):
    """Perform checks on a positional group node."""
    _, local_ctx = self.context
    if node.spec is None:
        raise InternalError('Missing node.spec for {}'.format(node))
    min_npargs = get_min_npargs(node.spec.npargs)
    semantic_tokens = node.get_semantic_tokens()
    if len(semantic_tokens) < min_npargs:
        location = ()
        if semantic_tokens:
            location = semantic_tokens[0].get_location()
        local_ctx.record_lint('E1120', location=location)
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
def parse_pragmas_from_comment(self, node):
    """Parse any cmake-lint directives (pragmas) from line comment tokens within
      the comment node."""
    suppressions = []
    for child in node.children:
        if not isinstance(child, Token):
            continue
        if child.type is not TokenType.COMMENT:
            continue
        token = child
        pragma_prefix = '# cmake-lint: '
        if not token.spelling.startswith(pragma_prefix):
            continue
        row, col, _ = token.get_location()
        content = token.spelling[len(pragma_prefix):]
        col += len(pragma_prefix)
        self.parse_pragmas_from_token(content, row, col, suppressions)
    return suppressions
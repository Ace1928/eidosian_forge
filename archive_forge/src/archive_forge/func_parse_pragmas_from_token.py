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
def parse_pragmas_from_token(self, content, row, col, suppressions):
    """Parse any cmake-lint directives (pragmas) from line-comment
      tokens at the current scope."""
    _, local_ctx = self.context
    items = content.split()
    for item in items:
        if '=' not in item:
            local_ctx.record_lint('E0011', item, location=(row, col))
            col += len(item) + 1
            continue
        key, value = item.split('=', 1)
        if key == 'disable':
            idlist = value.split(',')
            for idstr in idlist:
                if local_ctx.is_idstr(idstr):
                    suppressions.append(idstr)
                else:
                    local_ctx.record_lint('E0012', idstr, location=(row, col))
        else:
            local_ctx.record_lint('E0011', key, location=(row, col))
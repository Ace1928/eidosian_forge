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
def check_token(self, token):
    cfg, local_ctx = self.context
    if token.type is TokenType.WHITESPACE and token.get_location().col == 0:
        self._indent_token = token
        return
    if self._indent_token:
        desired_indentation = (self.get_scope_depth() - 1) * cfg.format.tab_size
        actual_indentation = len(self._indent_token.spelling) + self._indent_token.spelling.count('\t') * (cfg.format.tab_size - 1)
        if self.am_in_arggroup():
            if actual_indentation < desired_indentation:
                local_ctx.record_lint('C0307', self._indent_token.spelling, token.spelling, ' ' * desired_indentation, '>', location=token.get_location())
        elif actual_indentation != desired_indentation:
            msg = '->'.join([str(x) for x in self._node_stack])
            local_ctx.record_lint('C0307', self._indent_token.spelling, token.spelling, ' ' * desired_indentation, msg, location=token.get_location())
        self._indent_token = None
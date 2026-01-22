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
def loop_contains_argn(loop_stmt):
    """Return true if the loop statement contains ${ARGN} as an argument"""
    for token in loop_stmt.argtree.get_semantic_tokens():
        if token.type is TokenType.DEREF and token.spelling == '${ARGN}':
            return True
    return False
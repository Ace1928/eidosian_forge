from __future__ import print_function
from __future__ import unicode_literals
import logging
from cmakelang import lex
from cmakelang.common import UserError
from cmakelang.parse.util import COMMENT_TOKENS, WHITESPACE_TOKENS
from cmakelang.parse.common import NodeType, ParenBreaker, TreeNode
from cmakelang.parse.printer import tree_string
from cmakelang.parse.argument_nodes import StandardParser, StandardParser2
from cmakelang.parse.simple_nodes import CommentNode
def get_funname(self):
    return self.funnode.token.spelling.lower()